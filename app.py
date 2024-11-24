import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from PIL import Image
import streamlit as st
from matplotlib import pyplot as plt

# Define the VQA Model class
class VQAModel1(nn.Module):
    def __init__(self, num_answers):
        super(VQAModel1, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the final classification layer
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc1 = nn.Linear(2048 + 768, 1024)
        self.fc2 = nn.Linear(1024, num_answers)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images, input_ids, attention_mask):
        image_features = self.cnn(images)
        outputs = self.bert(input_ids, attention_mask)
        question_features = outputs.last_hidden_state[:, 0, :]
        combined_features = torch.cat((image_features, question_features), dim=1)
        x = self.fc1(combined_features)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQAModel1(num_answers=582)
checkpoint = torch.load("./best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Tokenizer and transforms
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load answer space
with open("./answer_space.txt") as f:
    answer_space = f.read().splitlines()

# Preprocess answer
def preprocess_answer(answer):
    return answer.replace("_", " ")

# Streamlit interface
st.title("Visual Question Answering")
st.write("Upload an image, ask a question, and get an AI-powered answer.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Input question
    question = st.text_input("Ask a question about the image:")
    if question:
        # Preprocess image
        image_tensor = image_transforms(image).unsqueeze(0).to(device)

        # Tokenize question
        inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=50)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor, input_ids, attention_mask)
            _, predicted_idx = torch.max(outputs, dim=1)
            predicted_answer = preprocess_answer(answer_space[predicted_idx.item()])

        # Display result
        st.write(f"Predicted Answer: **{predicted_answer}**")
