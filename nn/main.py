from datafetcher import viewpointCoordinateFinder, cleanData, generateUrlToRating
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import requests
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

# only uncomment when finding new coordinates TAKES ALOT OF API CALLS (unlimited just takes a while THANK YOU GOOGLE)
# viewpointCoordinateFinder()

# cleanData("california_viewpoints.csv")
# generateUrlToRating("california_viewpoints_cleaned.csv")


# ==========================
# ✅ 1. Define PyTorch Dataset (Loads Images from URLs)
# ==========================
class ViewpointDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)  # Load CSV with image URLs & ratings
        self.transform = transform  # Image transformations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_url = self.data.iloc[idx, 0]  # First column: Image URL
        rating = self.data.iloc[idx, 1]  # Second column: Rating

        # Fetch image from URL
        try:
            response = requests.get(image_url, timeout=10)  # Fetch image
            image = Image.open(BytesIO(response.content)).convert("RGB")  # Open as RGB
        except Exception as e:
            print(f"❌ Error loading image {image_url}: {e}")
            return None  # Skip if image cannot be loaded

        # Apply image transformations
        if self.transform:
            image = self.transform(image)

        # Convert rating to tensor
        rating = torch.tensor(float(rating), dtype=torch.float32)

        return image, rating  # Return (image, label) pair

# ==========================
# ✅ 2. Define Image Transformations
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================
# ✅ 3. Create Dataset & DataLoader
# ==========================

dataset = ViewpointDataset("urlToRatingWithoutNum.csv", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Example: Fetch one batch
images, ratings = next(iter(dataloader))
print(f"✅ Loaded batch of {len(images)} images")

# ==========================
# ✅ 4. Define Neural Network (ResNet for Regression)
# ==========================
class ViewpointScorer(nn.Module):
    def __init__(self):
        super(ViewpointScorer, self).__init__()
        self.base_model = models.resnet18(pretrained=True)  # Load pretrained ResNet
        self.base_model.fc = nn.Linear(512, 1)  # Change final layer for regression

    def forward(self, x):
        return self.base_model(x)

# Create model
model = ViewpointScorer()

# ==========================
# ✅ 5. Train the Model
# ==========================
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
counter = 0
for epoch in range(num_epochs):
    for images, ratings in dataloader:
        optimizer.zero_grad()
        predictions = model(images).squeeze(1)  # Remove extra dimension
        loss = criterion(predictions, ratings)  # Compute loss
        loss.backward()
        optimizer.step()
        print(f"{counter} step in epoch {epoch + 1}")
        counter += 1

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("✅ Training Complete!")

