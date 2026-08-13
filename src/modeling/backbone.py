import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) para redução de dimensionalidade
    de dados transcriptômicos de alta dimensão (Backbone do Caso A).
    """
    def __init__(self, input_dim, hidden_dim=400, latent_dim=50):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # --- ENCODER ---
        # Comprime: Genes -> Hidden -> (Mu, LogVar)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- DECODER ---
        # Reconstrói: Latent -> Hidden -> Genes
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        """Truque de reparametrização para permitir backprop através da estocasticidade."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  # Assumindo dados normalizados [0,1] ou usar Linear

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Função de Perda Estática (pode ser chamada sem instanciar)
    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        """
        Calcula a perda do VAE: Reconstrução (MSE) + Divergência KL.
        """
        # 1. Erro de Reconstrução
        BCE = F.mse_loss(recon_x, x.view(-1, recon_x.size(1)), reduction='sum')

        # 2. Divergência Kullback-Leibler (Regularização)
        # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD