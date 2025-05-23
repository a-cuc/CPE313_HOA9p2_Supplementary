import streamlit as st
import torch
from torchvision import models, transforms
from torch import nn
import cv2
from PIL import Image,ImageOps
import numpy as np

device = "cpu"
print(f"Using {device} device")

torch.classes.__path__ = []

from torch.nn.utils.parametrizations import weight_norm

class NeuralNetwork_Hyper(nn.Module):
    def __init__(self, cnn_l1 = 64, lstm_h = 64):
        super().__init__()
        self.cnn_layer = nn.Sequential(
            weight_norm(nn.Conv2d(3, cnn_l1, 3, bias=False)),
            nn.GroupNorm(8, cnn_l1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            weight_norm(nn.Conv2d(cnn_l1, 32, 3, bias=False)),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            weight_norm(nn.Conv2d(32, 32, 3, bias=False)),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(3),
            weight_norm(nn.Conv2d(32, 16, 3, bias=False)),
            nn.GroupNorm(8, 16),
            nn.ReLU(),
            weight_norm(nn.Conv2d(16, 16, 3, bias=False)),
            nn.GroupNorm(8, 16),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),

        )
        self.lstm_layer = nn.LSTM(input_size=16, hidden_size=lstm_h, num_layers=2,
                                  dropout=0.2, batch_first=True)
        self.fc_layer = nn.Linear(lstm_h, 1)

    def forward(self, x):
        x = self.cnn_layer(x)

        # Reshape for LSTM input
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, -1, channels)

        output, (h_n, c_n) = self.lstm_layer(x)
        logits = self.fc_layer(h_n[-1])
        return logits.squeeze(1)

model = NeuralNetwork_Hyper().to(device)
print(model)

st.write("""
# Drowsiness Detection System by Angelo"""
)
file=st.file_uploader("Choose photo from computer",type=["jpg","png"])

def import_and_predict(image_data,model):
    model.eval()
    transform = transforms.Compose([transforms.PILToTensor()])
    tensor = transform(image)
    with torch.no_grad():
        prediction=model(tensor)
    predicted_classes = (prediction > 0).float()
    return predicted_classes

if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Drowsy' 'Normal']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
