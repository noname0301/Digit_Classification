import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model():
    # Training parameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # Define transformations for the input data
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # mean and std for MNIST
    ])

    # Load the MNIST dataset
    data_dir = os.path.join(script_dir, "data")
    
    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):  # Just 5 epochs for simplicity
        epoch_loss = 0
        model.train()

        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]: {100. * i / len(train_loader):.0f}%, Loss: {loss.item():.4f}')

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        print(f'==> Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}')

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        accuracy = 100 * correct / total
        print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
    # Save the model

    # Build a path to save the file in that same folder
    file_path = os.path.join(script_dir, "mnist_model.pth")
    torch.save(model.state_dict(), file_path)
    print("Model saved to mnist_model.pth")

    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.show()
        

if __name__ == "__main__":
    model = LeNet5()
    print("Num of parameters: ", sum(p.numel() for p in model.parameters()))
    train_model()
