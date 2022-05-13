import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist

import time

# latent_dims = 2
# num_epochs = 50
# batch_size = 128
# capacity = 64
# learning_rate = 1e-3
# variational_beta = 1
# use_gpu = True

img_width = 28 
img_height = 28
img_size = img_width*img_height

class Encoder(nn.Module):
    def __init__(self, in_features, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.fc_1 = nn.Linear(in_features=in_features, out_features=hidden_size)
        self.fc_mu = nn.Linear(in_features=hidden_size, out_features=latent_size)
        self.fc_logvar = nn.Linear(in_features=hidden_size, out_features=latent_size)
        
    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        
        return x_mu, x_logvar
        

class Decoder(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super(Decoder, self).__init__()
        self.out_features = out_features
        self.fc_1 = nn.Linear(in_features=in_features, out_features=hidden_size)
        self.fc_2 = nn.Linear(in_features=hidden_size, out_features=img_size)
        
    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = torch.sigmoid(self.fc_2(x))
        return x
        
class MeanFieldAutoencoder(nn.Module):
    def __init__(self, in_features, latent_size):
        super(MeanFieldAutoencoder, self).__init__()
        self.encoder = Encoder(in_features=in_features, hidden_size=28,
                               latent_size=latent_size)
        # self.decoder = Decoder(in_size, 28, latent_size)
        self.decoder = Decoder(in_features=latent_size, hidden_size=28,
                               out_features=img_size)
        self.loss_mse = torch.nn.MSELoss(reduction='none')
        
    def forward(self, x):
        mu_latent, logvar_latent = self.encoder(x)
        
        std = logvar_latent.mul(0.5).exp_()
        latent = dist.normal.Normal(mu_latent, std).rsample()
        
        
        x_recon = self.decoder(latent)
        x_recon = x_recon.view(x_recon.shape[0], -1)
        return x_recon, mu_latent, logvar_latent
    
            
    
    def loss(self, recon_x, x, mu, logvar):
        # return torch.mean(torch.sum(self.loss_mse(recon_x, x)))
        
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        kl_div = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
        return recon_loss + 1 * kl_div # TODO: add variational beta?