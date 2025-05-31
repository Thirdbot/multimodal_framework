import os
# Set environment variables for better performance
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # Limit to single thread to avoid conflicts

import torch
from PIL import Image
from pathlib import Path
from chatTemplate import ChatTemplate
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_text_embeddings():
    # Initialize the template
    template = ChatTemplate()
    
    # Test texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fast orange fox leaps over a sleepy canine",
        "The weather is beautiful today",
        "It's raining cats and dogs"
    ]
    
    print("\nTesting Text Embeddings:")
    print("-" * 50)
    
    # Get embeddings for each text
    embeddings = []
    for text in texts:
        embedding = template.text_embedding(text)
        if embedding is not None:
            # Reshape the embedding to 2D if it's 3D
            if len(embedding.shape) == 3:
                embedding = embedding.reshape(embedding.shape[0], -1)
            embeddings.append(embedding)
            print(f"\nText: {text}")
            print(f"Embedding shape: {embedding.shape}")
            print(f"Embedding stats - min: {embedding.min():.4f}, max: {embedding.max():.4f}, mean: {embedding.mean():.4f}")
    
    # Calculate similarities between texts
    if len(embeddings) > 1:
        print("\nText Similarities:")
        print("-" * 50)
        # Stack embeddings into a 2D array
        embeddings_array = np.vstack(embeddings)
        similarities = cosine_similarity(embeddings_array)
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                print(f"Similarity between '{texts[i][:30]}...' and '{texts[j][:30]}...': {similarities[i][j]:.4f}")

def test_image_embeddings():
    # Initialize the template
    template = ChatTemplate()
    
    # Get test images
    HomePath = Path(__file__).parent.parent.absolute()
    image_dir = HomePath / "repositories" / "datasets" / "test_images"
    
    if not image_dir.exists():
        print(f"\nImage directory not found: {image_dir}")
        return
    
    # Get all image files
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if not image_files:
        print(f"\nNo images found in {image_dir}")
        return
    
    print("\nTesting Image Embeddings:")
    print("-" * 50)
    
    # Get embeddings for each image
    embeddings = []
    for img_path in image_files:
        try:
            image = Image.open(img_path).convert("RGB")
            embedding = template.image_embedding(image)
            if embedding is not None:
                embeddings.append(embedding)
                print(f"\nImage: {img_path.name}")
                print(f"Embedding shape: {embedding.shape}")
                print(f"Embedding stats - min: {embedding.min():.4f}, max: {embedding.max():.4f}, mean: {embedding.mean():.4f}")
        except Exception as e:
            print(f"Error processing image {img_path.name}: {str(e)}")
    
    # Calculate similarities between images
    if len(embeddings) > 1:
        print("\nImage Similarities:")
        print("-" * 50)
        similarities = cosine_similarity(embeddings)
        for i in range(len(image_files)):
            for j in range(i+1, len(image_files)):
                print(f"Similarity between '{image_files[i].name}' and '{image_files[j].name}': {similarities[i][j]:.4f}")

def test_cross_modal_similarity():
    # Initialize the template
    template = ChatTemplate()
    
    # Test text
    text = "A cat sitting on a windowsill"
    
    # Get test image
    HomePath = Path(__file__).parent.parent.absolute()
    image_dir = HomePath / "repositories" / "datasets" / "test_images"
    
    if not image_dir.exists():
        print(f"\nImage directory not found: {image_dir}")
        return
    
    # Get first image file
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if not image_files:
        print(f"\nNo images found in {image_dir}")
        return
    
    print("\nTesting Cross-Modal Similarity:")
    print("-" * 50)
    
    # Get text embedding
    text_embedding = template.text_embedding(text)
    if text_embedding is None:
        print("Failed to get text embedding")
        return
    
    print(f"\nText: {text}")
    print(f"Text embedding shape: {text_embedding.shape}")
    
    # Get image embedding
    try:
        image = Image.open(image_files[0]).convert("RGB")
        image_embedding = template.image_embedding(image)
        if image_embedding is None:
            print("Failed to get image embedding")
            return
        
        print(f"\nImage: {image_files[0].name}")
        print(f"Image embedding shape: {image_embedding.shape}")
        
        # Calculate similarity
        similarity = cosine_similarity(text_embedding, image_embedding)[0][0]
        print(f"\nCross-modal similarity: {similarity:.4f}")
        
    except Exception as e:
        print(f"Error processing image {image_files[0].name}: {str(e)}")

if __name__ == "__main__":
    print("Starting embedding tests...")
    
    # Test text embeddings
    test_text_embeddings()
    
    # Test image embeddings
    test_image_embeddings()
    
    # Test cross-modal similarity
    test_cross_modal_similarity()
    
    print("\nAll tests completed!")
