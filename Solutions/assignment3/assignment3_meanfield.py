from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
import torch
import matplotlib.pyplot as plt

from model_meanfield import MeanFieldAutoencoder
import utils


def train_meanfield_autoencoder(train_dataset, latent_dim, batch_size, num_epochs, learning_rate):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ## Define model
    feature_dim = train_dataset.data.shape[1] * train_dataset.data.shape[-1]
    model = MeanFieldAutoencoder(feature_dim=feature_dim, latent_dim=latent_dim)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-5)

    hist_loss = []
    print('Training ...')
    for epoch in range(num_epochs):
        hist_epoch_loss = []
        for image_batch, _ in train_dataloader:
            image_batch = image_batch.view(image_batch.shape[0], -1)

            # vae reconstruction
            image_batch_recon, latent_mu, latent_logvar = model(image_batch)

            # reconstruction error
            loss = model.loss(target=image_batch, pred=image_batch_recon)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()

            hist_epoch_loss.append(loss.item())

        hist_loss.append(torch.mean(torch.tensor(hist_epoch_loss)))
        print(f'Epoch [{epoch + 1} / {num_epochs}] average reconstruction error: {hist_loss[-1]}')

    return model, hist_loss


def main():
    latent_dim = 2
    num_epochs = 100
    batch_size = 128
    learning_rate = 1e-3

    # Train meanfield autoencoder on MNIST
    # load mnist dataset
    train_dataset = MNIST(root='./data',
                          download=True,
                          train=True,
                          transform=utils.img_transform)

    # train model
    model, hist_loss = train_meanfield_autoencoder(latent_dim=latent_dim,
                                                  batch_size=batch_size,
                                                  train_dataset=train_dataset,
                                                  num_epochs=num_epochs,
                                                  learning_rate=learning_rate)

    plt.plot(hist_loss)
    plt.title('Meanfield Autoencoder on MNIST')
    plt.savefig('./loss_meanfield_mnist.png')
    plt.show()
    torch.save(model, './model_meanfield_mnist.pt')

    # Train meanfield autoencoder on FashionMNIST
    # load mnist dataset
    train_dataset = FashionMNIST(root='./data',
                          download=True,
                          train=True,
                          transform=utils.img_transform)

    # train model
    model, hist_loss = train_meanfield_autoencoder(latent_dim=latent_dim,
                                                   batch_size=batch_size,
                                                   train_dataset=train_dataset,
                                                   num_epochs=num_epochs,
                                                   learning_rate=learning_rate)

    plt.plot(hist_loss)
    plt.title('Meanfield Autoencoder on FashionMNIST')
    plt.savefig('./loss_meanfield_fashionMnist.png')
    plt.show()
    torch.save(model, './model_meanfield_fashionMnist.pt')


if __name__ == "__main__":
    main()
