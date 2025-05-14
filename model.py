# model.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import random

# Dummy category list from the dataset (can be refined later)
CATEGORIES = ['Tshirt', 'Dress', 'Jeans', 'Topwear', 'Shoes', 'Shirt', 'Kurta', 'Jacket']

# Image pre-processing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Mock prediction function
def predict_outfit_score(image: Image.Image):
    # Apply transformation
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # ⚠️ You can load a trained model here later
    # For now, return a random score and category
    score = round(random.uniform(6.0, 10.0), 1)  # Simulated fashion score
    category = random.choice(CATEGORIES)

    return score, category
