from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
import torch
import matplotlib.pyplot as plt

import utils


def plot_reconstructions(dataloader, model, number_of_reconstructions, plot_title):
    inputs, classes = next(iter(dataloader))

    for image, label in zip(inputs[:number_of_reconstructions], classes[:number_of_reconstructions]):
        with torch.no_grad():
            pred, _, _ = model(image.view(image.shape[0], -1))
            pred = pred.squeeze().view(28, 28)
            image = image.view(28, 28)

        fig, ax = plt.subplots(1, 2)
        fig.tight_layout(pad=0.1)
        fig.suptitle(plot_title)

        ax[0].imshow(image)
        ax[0].set_title("original")

        ax[1].imshow(pred)
        ax[1].set_title(f"reconstruction | label={label}")

        plt.show()
        plt.close()


def main():
    batch_size = 128
    number_of_reconstructions = 5

    # load test data
    test_dataset = MNIST(root='./data',
                         download=True,
                         train=False,
                         transform=utils.img_transform)
    test_dataloader_mnist = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = FashionMNIST(root='./data',
                                download=True,
                                train=False,
                                transform=utils.img_transform)
    test_dataloader_fashionMnist = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # plot reconstructions for meanfield
    plot_reconstructions(dataloader=test_dataloader_mnist,
                         model=torch.load('./model_meanfield_mnist.pt'),
                         number_of_reconstructions=number_of_reconstructions,
                         plot_title='reconstructions for meanfield VAE')

    plot_reconstructions(dataloader=test_dataloader_fashionMnist,
                         model=torch.load('./model_meanfield_fashionMnist.pt'),
                         number_of_reconstructions=number_of_reconstructions,
                         plot_title='reconstructions for meanfield VAE')

    # plot reconstructions for covariance
    plot_reconstructions(dataloader=test_dataloader_mnist,
                         model=torch.load('./model_covariance_mnist.pt'),
                         number_of_reconstructions=number_of_reconstructions,
                         plot_title='reconstructions for covariance VAE')

    plot_reconstructions(dataloader=test_dataloader_fashionMnist,
                         model=torch.load('./model_covariance_fashionMnist.pt'),
                         number_of_reconstructions=number_of_reconstructions,
                         plot_title='reconstructions for covariance VAE')


if __name__ == "__main__":
    main()
