import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

mnist_mean = 0.1307
mnist_std = 0.3081  

# Define the transformation
transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(), 
    transforms.Normalize((mnist_mean,), (mnist_std,))])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Create a DataLoader to iterate through the dataset
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Visualize the first 5 images
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Plot the images in the batch
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    ax = axes[i]
    ax.imshow(images[i].squeeze(), cmap="gray")  # Squeeze removes the singleton dimension
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis('off')

plt.show()