import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision import models

# 1. Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Adjust for the number of classes in your model
    model.load_state_dict(torch.load('skin_disease_model.pth', map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    return model

# 2. Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),                      # Resize image
        transforms.ColorJitter(brightness=0.5),             # Increase brightness
        transforms.ToTensor(),                              # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],    # Normalize based on pre-trained model
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# 3. Prediction function
def predict(image, model):
    image = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# 4. Define class names (adjust according to your model's classes)
class_names = ['Benign', 'Malignant']

# 5. Streamlit UI
st.title("Skin Disease Prediction App")
st.write("Upload an image of a skin lesion and the app will predict the type of skin disease.")

# 6. Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Load the model
    model = load_model()

    # Predict the class
    prediction = predict(image, model)

    # Display the result
    st.write(f"Prediction: **{class_names[prediction]}**")