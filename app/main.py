import streamlit as st
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# Define DenseNet121Custom model
class DenseNet121Custom(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.3):
        super(DenseNet121Custom, self).__init__()
        # Load pre-trained DenseNet121
        self.densenet = models.densenet121(pretrained=True)

        # Remove the final classifier layer and add custom layers
        self.densenet.classifier = nn.Identity()  # Remove the final fully connected layer

        # Add a dropout layer and a new fully connected layer for classification
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1024, num_classes)  # DenseNet-121 outputs 1024 features

    def forward(self, x):
        # Forward pass through DenseNet-121 feature extractor
        x = self.densenet.features(x)
        x = torch.flatten(x, 1)  # Flatten the output
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)  # Final classification layer
        return x

# 1. Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = DenseNet121Custom(num_classes=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('model/skin_disease_detection_densenet121.pth', map_location=device))
    model.eval()
    return model, device

# 2. Image Preprocessing
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

# 3. Prediction function
def predict(image, model, device):
    image = preprocess_image(image)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

st.cache_data.clear()
st.cache_resource.clear()
# 4. Define class names (adjust according to your model's classes)
class_names = ['Actinic Keratosis', 'Melanoma', 'Squamous Cell Carcinoma', 'Basal Cell Carcinoma']

# 5. Streamlit UI
st.title("Skin Disease Prediction App")
st.write("Upload an image of a skin lesion and the app will predict the type of skin disease.")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stAppToolbar {visibility: hidden;}
            .stAlert {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 6. Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Load the model
    model, device = load_model()

    # Predict the class
    prediction = predict(image, model, device)

    # Display the result
    st.write(f"Prediction: **{class_names[prediction]}**")

st.cache_data.clear()
st.cache_resource.clear()