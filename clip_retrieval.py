import os
import pandas as pd
import clip
import torch
from PIL import Image
from torch import nn
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define paths to data
dataset_path = "../T3_AI/test_pepeganga"
image_folder = os.path.join(dataset_path, "test")
csv_file = os.path.join(dataset_path, "test.csv")

# Load the metadata CSV file
data = pd.read_csv(csv_file, sep=";")
print("Dataset loaded successfully.")
print(data.head())  # Verify the loaded data structure

# Cosine similarity function
cosine_similarity = nn.CosineSimilarity(dim=-1)

def encode_image(image_path):
    """Encodes an image using CLIP's visual encoder."""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features

def encode_text(text, max_length=77):
    """Encodes a text description using CLIP's text encoder."""
    truncated_text = text[:max_length]  # Truncate text to fit within CLIP's context length limit
    text_tokenized = clip.tokenize([truncated_text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokenized)
    return text_features

def retrieve_products(query_features, comparison_features, comparison_labels, top_k=5):
    """Retrieves top-k most similar products based on cosine similarity."""
    similarities = cosine_similarity(query_features, comparison_features)
    top_k_indices = similarities.topk(k=top_k).indices
    return comparison_labels.iloc[top_k_indices.cpu()]

def calculate_map(y_true, y_scores):
    """Calculates the mean average precision (mAP)."""
    return average_precision_score(y_true, y_scores)

def plot_recall_precision_curve(y_true, y_scores, title):
    """Plots Recall-Precision curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Recall-Precision Curve: {title}')
    plt.show()

def load_and_prepare_data():
    """Load dataset and prepare image and text features."""
    print("Loading dataset and preparing for retrieval...")

    # Prepare lists for image features and text features
    image_features_list = []
    text_features_list = []
    labels = data['GlobalCategoryEN']

    # Encode all images and descriptions
    for idx, row in data.iterrows():
        # Construct the image path using the correct column for the image filename
        image_path = os.path.join(image_folder, f"{row['Id']}.jpg")
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}, skipping this product.")
            continue  # Skip missing images

        # Process the image and text if the image file exists
        try:
            print(f"Image path for {row['ProductDescriptionEN']}: {image_path}")
            image_features = encode_image(image_path)
            text_features = encode_text(row['ProductDescriptionEN'])  # Truncate long text
            image_features_list.append(image_features)
            text_features_list.append(text_features)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Stack features into tensors
    if len(image_features_list) == 0 or len(text_features_list) == 0:
        print("No features were extracted, check your dataset.")
        return None, None, None

    image_features_tensor = torch.stack(image_features_list).squeeze(1)
    text_features_tensor = torch.stack(text_features_list).squeeze(1)
    
    print("Dataset preparation completed.")
    return image_features_tensor, text_features_tensor, labels

def main():
    # Load dataset and prepare features
    image_features_tensor, text_features_tensor, labels = load_and_prepare_data()
    
    if image_features_tensor is None or text_features_tensor is None:
        return
    
    print("Starting feature extraction...")

    # Now calculate the mAP and Recall-Precision (RP) for image and text retrieval
    print("Calculating mAP and plotting Recall-Precision curves...")

    # Image retrieval
    mAP_image = 0
    y_true_image = []
    y_scores_image = []
    
    # Loop through each product and retrieve similar products
    for i in range(min(len(data), len(image_features_tensor))):
        query_image_features = image_features_tensor[i]
        query_label = labels[i]
        
        # Calculate similarities for image retrieval
        similarities = cosine_similarity(query_image_features.unsqueeze(0), image_features_tensor)
        
        # Ground truth: products with the same category as the query
        y_true = (labels == query_label).astype(int)
        y_true_image.extend(y_true.tolist())
        y_scores_image.extend(similarities.tolist())
        
        # Example for retrieval:
        if i < 5:  # Retrieve top-5 products for first 5 queries as examples, gonna use this in the report
            similar_products = retrieve_products(query_image_features.unsqueeze(0), image_features_tensor, labels, top_k=5)
            print(f"\nExample {i+1} - Top 5 retrieved products for query image {i}:")
            print(similar_products)
    
    # Calculate mAP for image retrieval
    mAP_image = calculate_map(y_true_image, y_scores_image)
    print(f"Mean Average Precision (mAP) for image retrieval: {mAP_image}")

    # Plot RP curve for image retrieval
    plot_recall_precision_curve(y_true_image, y_scores_image, "Image-Based Retrieval")

    # Text retrieval (repeat for text descriptions)
    mAP_text = 0
    y_true_text = []
    y_scores_text = []
    
    # Loop through each product and retrieve similar products based on text
    for i in range(min(len(data), len(text_features_tensor))):
        query_text_features = text_features_tensor[i]
        query_label = labels[i]
        
        # Calculate similarities for text retrieval
        similarities = cosine_similarity(query_text_features.unsqueeze(0), text_features_tensor)
        
        # Ground truth: products with the same category as the query
        y_true = (labels == query_label).astype(int)
        y_true_text.extend(y_true.tolist())
        y_scores_text.extend(similarities.tolist())
    
    # Calculate mAP for text retrieval
    mAP_text = calculate_map(y_true_text, y_scores_text)
    print(f"Mean Average Precision (mAP) for text retrieval: {mAP_text}")

    # Plot RP curve for text retrieval
    plot_recall_precision_curve(y_true_text, y_scores_text, "Text-Based Retrieval")

if __name__ == "__main__":
    main()
