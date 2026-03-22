import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

# EarlyStopping class from notebook
class EarlyStopping:
    def __init__(self, patience=16, verbose=True, delta=0):
        self.patience  = patience
        self.verbose   = verbose
        self.delta     = delta
        self.counter   = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            if self.verbose and self.best_loss is not None:
                print(f"✅ Loss decreased: {self.best_loss:.4f} -> {val_loss:.4f}")
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def pad_to_square(img, fill=0):
    w, h = img.size
    if w == h: return img
    diff = abs(h - w)
    pad1, pad2 = diff // 2, diff - diff // 2
    padding = (0, pad1, 0, pad2) if h < w else (pad1, 0, pad2, 0)
    return TF.pad(img, padding, fill=fill)

def get_model(num_classes: int, img_size: int):
    class CustomCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2)
            )
            # 128 * (img_size // 2^3) * (img_size // 2^3)
            # For 128x128, it's 128 * 16 * 16
            feature_dim = 128 * (img_size // 8) * (img_size // 8)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feature_dim, 256),
                nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        def forward(self, x):
            return self.classifier(self.features(x))
    return CustomCNN()

def train_model(bean_type: str, dataset_path: str, model_dir: str):
    log_file = os.path.join(model_dir, f"{bean_type}_training.log")
    with open(log_file, "w") as f:
        f.write(f"Starting training for {bean_type}...\n")
        f.flush()
    
    print(f"\n>>> Training for {bean_type}...")
    batch_size = 256
    img_size = 128
    num_epochs = 40 # Reduced from 80 to save time while still being effective
    lr = 5e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Lambda(lambda img: pad_to_square(img, fill=0)),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    if not os.path.exists(dataset_path):
        print(f"Skipping {bean_type}: Path {dataset_path} not found.")
        return

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    num_classes = len(dataset.classes)
    
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = get_model(num_classes, img_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopping(patience=10, verbose=True)

    for epoch in range(1, num_epochs+1):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                val_loss += criterion(model(imgs), lbls).item() * lbls.size(0)
        val_loss /= len(test_ds)
        
        print(f"Epoch {epoch}/{num_epochs}, Val Loss: {val_loss:.4f}")
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch}/{num_epochs}, Val Loss: {val_loss:.4f}\n")
            f.flush()
        
        stopper(val_loss, model)
        if stopper.early_stop:
            print("Early stopping triggered.")
            with open(log_file, "a") as f:
                f.write("Early stopping triggered.\n")
                f.flush()
            break

    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"{bean_type}_custom_Noback_best_model.pth")
    torch.save(stopper.best_model_state, save_path)
    print(f"Saved: {save_path}")
    with open(log_file, "a") as f:
        f.write(f"Saved: {save_path}\n")
        f.flush()

if __name__ == "__main__":
    beans = ["ethiopia_washed", "kenya_natural", "honduras_natural"]
    base_data_path = "coffee_beans_data"
    model_dir = "models"
    
    for bean in beans:
        path = os.path.join(base_data_path, bean, "corp_augmented_dataNoback")
        train_model(bean, path, model_dir)
