import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DVariationalEncoder_fa(nn.Module):
    def __init__(self, latent_dims=20, dropout=0.2, input_length=50):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2_50 = nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=2)
        self.conv2_100 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        # self.conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)

        # Calculate the output size dynamically
        self._dummy_input = torch.zeros(1, 1, input_length)
        self._conv_output = self._get_conv_output_shape(self._dummy_input)
        self.flattened_size = self._conv_output[1] * self._conv_output[2]
        
        self.fc_mean = nn.Linear(self.flattened_size, latent_dims)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dims)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def _get_conv_output_shape(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2_100(x))
        x = F.relu(self.conv3(x))
        return x.shape
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        # x = F.relu(self.conv2_50(x))
        x = F.relu(self.conv2_100(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        # x = F.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.flatten(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)    

        return mean, logvar

# Variational decoder for reconstructing flattened FA tract data from latent vectors
# Uses transposed convolutions to upsample latent code back to original sequence length
# Paired with Conv1DVariationalEncoder_fa to form complete VAE
class Conv1DVariationalDecoder_fa(nn.Module):
    def __init__(self, latent_dims=20, conv_output_shape=None):
        super().__init__()
        # Store the expected shape of the conv features
        self.conv_channels = conv_output_shape[1]  # 64
        self.conv_length = conv_output_shape[2]    # 7 for input_length=50, 14 for input_length=100
        self.flattened_size = self.conv_channels * self.conv_length
        
        self.fc = nn.Linear(latent_dims, 64 * 13)
        self.deconv2 = nn.ConvTranspose1d(self.conv_channels, 32, kernel_size=5, stride=2, padding=2, output_padding=0)
        self.deconv3_100 = nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3_50 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.fc(x)
        x = x.view(batch_size, 64, 13)
        x = F.relu(self.deconv2(x))
        # x = F.relu(self.deconv3_50(x))
        x = F.relu(self.deconv3_100(x))
        x = self.deconv4(x)
        return x

# Complete variational autoencoder for flattened FA tract data
# Combines encoder and decoder with reparameterization trick for stochastic latent sampling
# Designed for single-channel 1D sequences representing tract profiles
class Conv1DVariationalAutoencoder_fa(nn.Module):
    def __init__(self, latent_dims=20, dropout=0.0, input_length=50):
        super().__init__()
        self.encoder = Conv1DVariationalEncoder_fa(latent_dims, dropout=dropout, input_length=input_length)
        # Pass the shape information from encoder to decoder
        self.decoder = Conv1DVariationalDecoder_fa(latent_dims, self.encoder._conv_output)
        self.latent_dims = latent_dims
        
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_prime = self.decoder(z)
        return x_prime, mean, logvar
    
class Conv1DEncoder_fa(nn.Module):
    def __init__(self, latent_dims=20, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        
        # Instead of directly mapping to latent space, we'll produce two outputs:
        # mean and log variance (each of size latent_dims)
        self.conv4 = nn.Conv1d(64, latent_dims, kernel_size=5, stride=2, padding=2)
        # self.conv4_logvar = nn.Conv1d(64, latent_dims, kernel_size=5, stride=2, padding=2)

        # self.fc_mean = nn.Linear(64*7, latent_dims)
        # self.fc_logvar = nn.Linear(64*7, latent_dims)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x = torch.flatten(x, 1)
        x = F.relu(self.conv1(x)) # [64, 16, 25]
        x = self.dropout(x)
        x = F.relu(self.conv2(x)) # [64, 32, 13]
        x = self.dropout(x)
        x = F.relu(self.conv3(x)) # [64, 64, 7]
        x = self.dropout(x)
        x = self.conv4(x) # [64, 64, 4]

        return x

class Conv1DDecoder_fa(nn.Module):
    def __init__(self, latent_dims=20):
        super().__init__()
        # self.fc = nn.Linear(latent_dims, 64 * 7)
        self.deconv1 = nn.ConvTranspose1d(latent_dims, 64, kernel_size=5, stride=2, padding=2, output_padding=0)
        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=0)
        self.deconv3 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        return x

class Conv1DAutoencoder_fa(nn.Module):
    def __init__(self, latent_dims=20, dropout=0.0):
        super().__init__()
        self.encoder = Conv1DEncoder_fa(latent_dims, dropout=dropout)
        self.decoder = Conv1DDecoder_fa(latent_dims)
        self.latent_dims = latent_dims
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class AgePredictorCNN(nn.Module):
    def __init__(self, input_channels=1, sequence_length=50, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        _dummy_input = torch.randn(1, input_channels, sequence_length)
        _conv_output_shape = self._get_conv_output_shape(_dummy_input)
        flat_size = _conv_output_shape[1] * _conv_output_shape[2]

        self.fc1 = nn.Linear(flat_size, 64)
        self.fc_out = nn.Linear(64, 1)

    def _get_conv_output_shape(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x.shape

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        age_pred = self.fc_out(x)
        return age_pred

class SitePredictorCNN(nn.Module):
    def __init__(self, num_sites=4, input_channels=1, sequence_length=50, dropout=0.3):
        super().__init__()
        # Fewer initial filters and larger dropout
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        # Remove the third convolutional layer to make this network simpler
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)  # Higher dropout
        self.relu = nn.ReLU()
        
        dummy_input = torch.randn(1, input_channels, sequence_length)
        conv_output_shape = self._get_conv_output_shape(dummy_input)
        flat_size = conv_output_shape[1] * conv_output_shape[2]
        
        self.fc1 = nn.Linear(flat_size, 32)  # Smaller hidden layer
        self.fc_out = nn.Linear(32, num_sites)
        
    def _get_conv_output_shape(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x.shape
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        site_pred = self.fc_out(x)
        return site_pred

try:
    from .utils import grad_reverse
except ImportError:
    # Fallback if relative import fails (e.g., script run directly)
    try:
        from utils import grad_reverse # Assuming utils.py is in PYTHONPATH
    except ImportError:
        print("Warning: grad_reverse function not found. Define or import it for CombinedVAE_Predictors.")
        # Define a dummy grad_reverse if not found, so the class can be defined
        def grad_reverse(x, alpha=1.0):
            print("Warning: Using dummy grad_reverse!")
            return x


class CombinedVAE_Predictors(nn.Module):
    def __init__(self, vae_model, age_predictor, site_predictor):
        super().__init__()
        self.vae = vae_model
        self.age_predictor = age_predictor
        self.site_predictor = site_predictor

    def forward(self, x, grl_alpha=1.0):
        x_hat, mean, logvar = self.vae(x)
        age_pred = self.age_predictor(x_hat)
        x_hat_reversed = grad_reverse(x_hat, grl_alpha)
        site_pred = self.site_predictor(x_hat_reversed)
        return x_hat, mean, logvar, age_pred, site_pred