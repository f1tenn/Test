import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image

def prepare_data(data_dir):
    train_paths = []
    val_paths = []
    test_paths = []

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        for brand in ['Bialetti', 'De\'Longhi']:
            brand_dir = os.path.join(split_dir, brand)
            for filename in os.listdir(brand_dir):
                if filename.startswith('image_') and filename.endswith('.jpg'):
                    file_path = os.path.join(brand_dir, filename)
                    if split == 'train':
                        train_paths.append(file_path)
                    elif split == 'val':
                        val_paths.append(file_path)
                    elif split == 'test':
                        test_paths.append(file_path)
    
    return train_paths, val_paths, test_paths

def train_model(train_paths, val_paths, model_save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_dataset = ImageDataset(train_paths)
    val_dataset = ImageDataset(val_paths)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            outputs = model.get_image_features(pixel_values=batch)
            loss = criterion(outputs, torch.zeros(outputs.size(0), dtype=torch.long, device=device))
            
            if loss is None:
                raise ValueError("Loss calculation failed. Ensure the model and data are properly set up.")
            
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model.get_image_features(pixel_values=batch)
                loss = criterion(outputs, torch.zeros(outputs.size(0), dtype=torch.long, device=device))
                val_loss += loss.item()
            
            print(f"Validation Loss after epoch {epoch+1}: {val_loss / len(val_loader)}")

    # Save
    torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    data_dir = 'data'
    model_save_path = 'model/model.pt'
    train_paths, val_paths, test_paths = prepare_data(data_dir)

    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    print(f"Test set: {len(test_paths)} images")

    train_model(train_paths, val_paths, model_save_path)
