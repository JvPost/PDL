import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim):
        super(Encoder, self).__init__()
        self.fc_input = nn.Linear(in_features=in_dim, out_features=hidden_size)
        self.fc_mu = nn.Linear(in_features=hidden_size, out_features=out_dim)
        self.fc_logvar = nn.Linear(in_features=hidden_size, out_features=out_dim**2)

    def forward(self, x):
        y = F.relu(self.fc_input(x))
        mu = self.fc_mu(y)
        var = F.softplus(self.fc_logvar(y))
        # compute the lower triangular matrix
        var = torch.tril(var).view(-1, 2, 2)

        return mu, var


class Decoder(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim):
        super(Decoder, self).__init__()
        self.fc_input = nn.Linear(in_features=in_dim, out_features=hidden_size)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=out_dim)

    def forward(self, x):
        y = F.relu(self.fc_input(x))
        out = torch.sigmoid(self.fc_out(y))
        return out


class CovarianceAutoencoder(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super(CovarianceAutoencoder, self).__init__()
        self.encoder = Encoder(in_dim=feature_dim, hidden_size=50, out_dim=latent_dim)
        self.decoder = Decoder(in_dim=latent_dim, hidden_size=50, out_dim=feature_dim)
        self.loss_mse = torch.nn.MSELoss(reduction='none')

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        # latent = dist.normal.Normal(latent_mu, latent_logvar).rsample()
        # print(latent_mu.shape, latent_logvar.shape)
        latent = dist.multivariate_normal.MultivariateNormal(latent_mu, scale_tril=latent_logvar).rsample()
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def loss(self, pred, target):
        return torch.mean(torch.sum(self.loss_mse(pred, target), dim=-1))
