import torch
from dataset import SuperSDataset
from torchvision import transforms as tf
from torch.utils.data import DataLoader
import os
from model import FaceUNet, FaceUNetLoss
import tqdm

torch.cuda.empty_cache()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

print(f"starting operation on device: {DEVICE}")

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


# paths
train_path = "E:/Projects/RealProjects/MachineLearningPorjects/FaceDetection/temp/backup/img_align_celeba/train"
valid_path = "E:/Projects/RealProjects/MachineLearningPorjects/FaceDetection/temp/backup/img_align_celeba/valid"

# transforms
transform = tf.Compose([
    tf.Resize([176, 216]),
    tf.ToTensor(),
])


# datasets
train_dataset = SuperSDataset(train_path, transform)
valid_dataset = SuperSDataset(valid_path, transform)

print(f"found {len(train_dataset)} training images")
print(f"found {len(valid_dataset)} validation images")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# model, loss, optimizer
model = FaceUNet().to(DEVICE)
epochs = 15
criterion = FaceUNetLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# early stopping
patience_number = 3
delta = 0.001
best_val_loss = float("inf")
patience = 0

# outputs
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/checkpoints", exist_ok=True)
train_losses = []
val_losses = []

print(f"🚀 Start training with {epochs} epochs ...")

for epoch in range(epochs):
    # train
    model.train()
    train_loss = 0.0
    print(train_loader.batch_size)
    print(len(train_loader.dataset))
    exit()
    
    for batch in tqdm.tqdm(train_loader, total=len(train_loader), desc="feeding batchs ..."):
        blrd_img, main_img = batch["blured"].to(DEVICE), batch["main"].to(DEVICE)
        
        optimizer.zero_grad()
        output = model(blrd_img)
        loss = criterion(output, main_img)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)
    
    # validation
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch in tqdm.tqdm(valid_loader, total=len(valid_loader), desc="validating model ..."):
            blrd_img, main_img = batch["blured"].to(DEVICE), batch["main"].to(DEVICE)
            
            output = model(blrd_img)    
            loss = criterion(output, main_img)
            valid_loss += loss.item()
        
    avg_val_loss = valid_loss / len(valid_loader)
    
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    # early stopping
    if avg_val_loss < best_val_loss - delta:
        best_val_loss = avg_val_loss
        patience = 0
        torch.save(model.state_dict(), "outputs/checkpoints/superres_best.pth")
        print(f"✅ Validation improved. Saving model.")
    else:
        patience += 1
        print(f"⚠️ No improvement. Early stop counter: {patience}/{patience_number}")
        if patience >= patience_number:
            print("⛔ Early stopping triggered!")
            break


import matplotlib.pyplot as plt


plot_losses(train_losses, val_losses)
