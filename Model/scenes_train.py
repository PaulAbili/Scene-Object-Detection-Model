import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pathlib import Path
from ultralytics import YOLO

from scenes_classifier import SceneClassifier

device = torch.device("cuda")

scene_model = YOLO("/umbc/class/cmsc475sp26/common/group_5/completed_runs/trained_objects/weights/best.pt")

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.Places365(
    root="/umbc/class/cmsc475sp26/common/group_5/scenes_data/places365",
    split="train-standard",
    small=True,
    transform=transform
)

val_dataset = datasets.Places365(
    root="/umbc/class/cmsc475sp26/common/group_5/scenes_data/places365",
    split="val",
    small=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
print(f"Classes: {len(train_dataset.classes)}")

def one_epoch(model, loader, optimizer, criterion, device, epoch, unfreeze):
    total_loss = 0
    correct = 0
    total = 0
    model.train()
    if epoch < unfreeze:
        model.backbone.eval()
        for layer in list(model.backbone)[:-1]:
            layer.eval()

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = (correct / total) * 100

    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    total_loss = 0
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            
            total_loss += loss.item()
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)
            
    avg_loss = total_loss / len(loader)
    accuracy = (correct / total) * 100
        
    return avg_loss, accuracy

def training_loop(epochs, train_loader, val_loader, device, unfreeze=20, resume=False):
    model = SceneClassifier(scene_model.model.model, num_classes=365, classes=train_dataset.classes).to(device)
    optimizer = optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0

    best_val_accuracy = 0.0
    save_dir = Path("/umbc/class/cmsc475sp26/common/group_5/completed_runs/trained_scenes/weights")
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    if resume:
        model = torch.load(save_dir/"last_scenes.pt", map_location=device)
        info = torch.load(save_dir/"last_info.pt", map_location=device)
        optimizer.load_state_dict(info["optimizer"])
        scheduler.load_state_dict(info["scheduler"])
        best_val_accuracy = info["best_val_accuracy"]
        start_epoch = info["epoch"]
        history = info["history"]

        if start_epoch >= unfreeze:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer.add_param_group({'params': model.backbone.parameters(), 'lr': 1e-4})
        print("Resuming Training")
    
    for epoch in range(start_epoch, epochs):
        start = time.time()

        print(f"Current Epoch: {epoch+1}/{epochs}")

        if epoch == unfreeze:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer.add_param_group({'params': model.backbone.parameters(), 'lr': 1e-4})
        
        train_loss, train_accuracy = one_epoch(model, train_loader, optimizer, criterion, device, epoch, unfreeze)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        scheduler.step()

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        
        elapsed = time.time() - start
        
        print(f"Epoch {epoch+1:3d}/{epochs}, Time: {elapsed:.1f}s, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(obj=model, f=save_dir/"best_scenes.pt")
            best_info = {
                "epoch": epoch + 1,
                "best_val_accuracy": best_val_accuracy,
                "history": history,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "classes": model.classes
            }
            torch.save(obj=best_info, f=save_dir/"best_scenes_info.pt")
            print(f"Saved model (Accuracy = {best_val_accuracy:2f}%)")

        torch.save(obj=model, f=save_dir/"last_scenes.pt")
        last = {
            "epoch": epoch + 1,
            "best_val_accuracy": best_val_accuracy,
            "history": history,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "classes": train_dataset.classes
        }
        torch.save(obj=last, f=save_dir/"last_info.pt")

training_loop(epochs=20, train_loader=train_loader, val_loader=val_loader, device=device, unfreeze=10)