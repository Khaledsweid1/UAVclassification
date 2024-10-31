import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import UAVClassifier

def train_model(data_dir, num_epochs=3, batch_size=32):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = UAVClassifier(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')
    print("Training complete! Model saved as 'model.pth'")

if __name__ == "__main__":
    data_dir = 'C:\\Users\\KHALED\\Downloads\\UAV_Classification\\data'  # Adjust this to your dataset path
    train_model(data_dir)
