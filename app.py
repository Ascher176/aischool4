import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

# copied from ai_school_3 notebook
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer: 3 input channels, 6 output channels, 3x3 kernel size
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        # Second convolutional layer: 6 input channels, 16 output channels, 3x3 kernel size
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Convolutional layer 1 + ReLU + Max pooling
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)

        # Convolutional layer 2 + ReLU + Max pooling
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)

        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 16 * 8 * 8)

        # Fully connected layers + ReLU
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Final output layer
        x = self.fc3(x)
        return x

# Load weights for our model
model = SimpleCNN()
model.load_state_dict(torch.load("simple_cnn_model.pth"))
model.eval()

# CIFAR-10 classes
class_labels = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 images are 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# App title and description
st.title("CIFAR-10 Image Classification")
st.write("Upload an image, and the model will classify it into one of the CIFAR-10 categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess the image
    st.write("Processing the image...")
    input_tensor = transform(image).unsqueeze(0)  # apply transformations

    # Put image through model
    with torch.no_grad():
        output = model(input_tensor) # get predictions

        # softmax to convert output into probabilities (now they sum up to 1)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get index of class with the highest probability
        confidence, class_idx = torch.max(probabilities, dim=0)

    # Class index to label
    predicted_label = class_labels[class_idx.item()]

    # Display the prediction and confidence
    st.write("**Predicted Class:**", predicted_label)
    st.write("**Confidence:**", round(confidence.item(),2))

    # Display Top 3 Predictions
    st.write("### Top 3 Predictions:")
    top3_indices = torch.topk(probabilities, 3).indices
    for idx in top3_indices:
        st.write("-", class_labels[idx.item()], ":", round(probabilities[idx].item(),2))