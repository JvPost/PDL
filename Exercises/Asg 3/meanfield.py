from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, MNIST
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from model_meanfield import MeanFieldAutoencoder

img_transform = transforms.Compose([
    transforms.ToTensor()
])

use_gpu = True
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
device

def train(training_dataset, latent_size, batch_size, num_epochs, learning_rate):
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle = True)
    
    feature_size = training_dataset.data.shape[1] * training_dataset.data.shape[-1]
    model = MeanFieldAutoencoder(feature_size, latent_size)
    model.to(device)
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    hist_loss = []
    print("training...")
    for epoch in range(num_epochs):
        hist_epoch_loss = []
        for image_batch, _ in train_dataloader:
            image_batch = image_batch.view(image_batch.shape[0], -1).to(device)
            
            img_batch_recon, mu_latent, logvar_latent = model(image_batch)
            loss = model.loss(img_batch_recon, image_batch, mu_latent, logvar_latent)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            hist_epoch_loss.append(loss.item())
        
        hist_loss.append(torch.mean(torch.tensor(hist_epoch_loss)))
        print(f'Epoch [{epoch + 1} / {num_epochs}] average reconstruction error: {hist_loss[-1]}')
        
    return model, hist_loss
            

def main():
    latent_size = 2
    num_epochs = 30
    batch_size=128
    learning_rate = 1e-3
    
    training_dataset = MNIST(root='~/data',
                          download=False,
                          train=True,
                          transform = img_transform)
    
    model, hist_loss = train(
        training_dataset=training_dataset,
        latent_size=latent_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )
    
    return

if __name__ == "__main__":
    main()