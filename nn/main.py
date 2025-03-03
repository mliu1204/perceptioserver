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

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = ViewpointDataset("urlToRatingWithoutNum.csv", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

class ViewpointScorer(nn.Module):
    def __init__(self):
        super(ViewpointScorer, self).__init__()
        self.base_model = models.resnet18(pretrained=True)  # Load pretrained ResNet
        self.base_model.fc = nn.Linear(512, 1)  # Change final layer for regression

    def forward(self, x):
        return self.base_model(x)

def train():
    # Create model
    model = ViewpointScorer()

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
    torch.save(model.state_dict(), "viewpoint_scorer.pth")
    print("Training Complete!")


def load_model():
    model = ViewpointScorer()
    model.load_state_dict(torch.load("viewpoint_scorer.pth"))
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully")
    return model

def preprocess_image(image_url):
    response = requests.get(image_url, timeout=10)
    image = Image.open(BytesIO(response.content)).convert("RGB")  # Convert to RGB

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match training input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform(image).unsqueeze(0)  # Add batch dimension

def predict_viewpoint_score(image_url, model):
    """Predict viewpoint score using trained model."""
    image_tensor = preprocess_image(image_url)

    with torch.no_grad():  # No gradients needed for inference
        score = model(image_tensor).item()  # Convert to float

    return score

image_url = "https://lh3.googleusercontent.com/place-photos/ADOriq2zpBHXHZadeqI__vOrKUj9rK-R7F-fKf5-h410kV_1GAZri556Rcp3-j_uoF7hcc1TKkf_DBVbcAOIwVtvZgTjuXhXXYoC1or8swqv_L4I_glXSN2mS8wdgH_-awT7rXsVUz8K=s1600-w800"

model = load_model()
predicted_score = predict_viewpoint_score(image_url, model)
print(f"Predicted viewpoint score: {predicted_score:.2f}")
