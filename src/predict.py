import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

def predict(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    outputs = model.get_image_features(**inputs)

    brand_predictions = ["Bialetti", "De'Longhi"]
    return brand_predictions[outputs.argmax()]

if __name__ == "__main__":
    image_path = " \data\test\De'Longhi\image_2.jpg"
    model_path = "model/model.pt"
    result = predict(image_path, model_path)
    print(f"Brand: {result}")
