import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split

# ==== Paths ====
images_dir = './images'
styles_file = './styles.csv'

# ==== Load and Clean CSV ====
df = pd.read_csv(styles_file, on_bad_lines='skip')

df = df.sample(n=1000, random_state=42)  # Use only 1000 samples


# Keep only images that actually exist
existing_images = set(f.split('.')[0] for f in os.listdir(images_dir))
df = df[df['id'].astype(str).isin(existing_images)]

# Convert category labels to numbers
df['masterCategory'] = df['masterCategory'].astype('category')
df['label'] = df['masterCategory'].cat.codes

# Split into train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ==== Transform ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # For RGB
])

# ==== Dataset ====
class FashionDataset(Dataset):
    def __init__(self, dataframe, images_dir, transform=None):
        self.dataframe = dataframe
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = str(self.dataframe.iloc[idx]['id'])
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        label = int(self.dataframe.iloc[idx]['label'])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# ==== Dataloaders ====
train_dataset = FashionDataset(train_df, images_dir, transform)
test_dataset = FashionDataset(test_df, images_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==== Model ====
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Updated way
num_classes = df['label'].nunique()
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ==== Loss & Optimizer ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==== Training ====
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# ==== Save Model ====
torch.save(model.state_dict(), 'fashion_model.pth')
print("✅ Model saved as fashion_model.pth")
