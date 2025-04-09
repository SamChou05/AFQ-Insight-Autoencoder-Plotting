import torch
import torch.nn.functional as F
from afqinsight.datasets import AFQDataset
from afqinsight.nn.utils import prep_pytorch_data
from torch.cuda.amp import autocast, GradScaler
import numpy as np

def get_beta(current_epoch, total_epochs, start_epoch=100):
    if current_epoch < start_epoch:
        return 0.0
    
    progress = (current_epoch - start_epoch) / (total_epochs - start_epoch)
    
    beta = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5)))

    return beta

def train_variational_autoencoder(model, train_data, val_data, epochs=500, lr=0.001, kl_weight=0.001, device = 'cuda'):
    """
    Training loop for variational autoencoder with KL annealing
    """
    torch.backends.cudnn.benchmark = True

    latent_dim = model.latent_dims if hasattr(model, 'latent_dims') else "unknown"
    dropout = model.dropout if hasattr(model, 'dropout') else "unknown"
    
    # Create a unique model filename
    model_filename = f"best_vae_model_ld{latent_dim}_dr{dropout}.pth"

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5)
    
    scaler = torch.amp.GradScaler(device=device)  # For mixed precision training

    train_rmse_per_epoch = []
    val_rmse_per_epoch = []
    train_kl_per_epoch = []
    val_kl_per_epoch = []   
    train_recon_per_epoch = []
    val_recon_per_epoch = []
    
    best_val_rmse = float('inf')  # Track the best (lowest) validation RMSE
    best_model_state = None  # Save the best model state
    best_epoch = 0

    #lets try KL annealing
    beta_start = 0.0
    beta_end = 1.0
    slope = (beta_end - beta_start) / epochs
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0
        running_rmse = 0
        running_kl = 0
        items = 0
        running_recon_loss = 0 
        # beta = beta_start + slope * epoch
        beta = get_beta(epoch, epochs)
        
        for x, _ in train_data:
            batch_size = x.size(0)
            tract_data = x.to(device, non_blocking=True)
            
            opt.zero_grad(set_to_none=True)
            
            # Forward pass returns reconstructed x, mean and logvar
            with torch.amp.autocast(device_type= "cuda"):
                x_hat, mean, logvar = model(tract_data)
                
                # Compute loss with KL divergence
                loss, recon_loss, kl_loss = vae_loss(tract_data, x_hat, mean, logvar, beta, reduction="sum")
                #recon loss here is the sum of the MSE
                #loss is the sum of the KL and the recon loss
                #kl loss is the sum of the KL
                #none are normalized yet 

                # Calculate RMSE (primarily for logging)
                batch_rmse = torch.sqrt(F.mse_loss(tract_data, x_hat, reduction="mean"))
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
              
            #increasing by batch size
            items += batch_size
            running_loss += loss.item()
            running_rmse += batch_rmse.item() * batch_size  # Weighted sum
            running_kl += kl_loss.item() # Average KL per item
            running_recon_loss += recon_loss.item() # Average recon loss per item
        
        scheduler.step(running_loss / items)
        avg_train_rmse = running_rmse / items
        avg_train_kl = running_kl / items 
        avg_train_recon_loss = running_recon_loss / items
        train_rmse_per_epoch.append(avg_train_rmse)
        train_kl_per_epoch.append(avg_train_kl)
        train_recon_per_epoch.append(avg_train_recon_loss)

        # Validation
        model.eval()
        val_rmse = 0
        val_kl = 0
        val_items = 0
        val_recon_loss = 0
        
        with torch.no_grad():
            for x, *_ in val_data:
                batch_size = x.size(0)
                tract_data = x.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type = "cuda"):

                    x_hat, mean, logvar = model(tract_data)
                    
                    val_loss, val_recon_loss, val_kl_loss = vae_loss(tract_data, x_hat, mean, logvar, beta, reduction="sum")
                    
                    batch_val_rmse = torch.sqrt(F.mse_loss(tract_data, x_hat, reduction="mean"))

                val_items += batch_size
                val_loss += val_loss.item()
                val_rmse += batch_val_rmse.item() * tract_data.size(0)
                val_kl += val_kl_loss.item()
                val_recon_loss += val_recon_loss.item()
        
        avg_val_recon_loss = val_recon_loss / val_items
        avg_val_rmse = val_rmse / val_items
        avg_val_kl = val_kl / val_items
        val_rmse_per_epoch.append(avg_val_rmse)
        val_kl_per_epoch.append(avg_val_kl)
        val_recon_per_epoch.append(avg_val_recon_loss)
        
        # Check and save the best model state if current validation loss is lower
        if avg_val_rmse < best_val_rmse:
            print(f"Saving best model state with RMSE: {avg_val_rmse:.4f} at epoch {epoch+1}")
            best_val_rmse = avg_val_rmse
            best_model_state = model.state_dict().copy()  # Make a copy to ensure it's preserved
            best_epoch = epoch + 1  # Make a copy to ensure it's preserved

            torch.save(best_model_state, model_filename)
            print(f"Best model saved to: {model_filename}")
        
        print(f"Epoch {epoch+1}, Train RMSE: {avg_train_rmse:.4f}, Val RMSE: {avg_val_rmse:.4f}, KL: {avg_train_kl:.4f}," ,
              f"Recon Loss (Train): {avg_train_recon_loss:.4f}, Recon Loss (Val): {avg_val_recon_loss:.4f}")
    
    return {
        "train_rmse_per_epoch": train_rmse_per_epoch,
        "val_rmse_per_epoch": val_rmse_per_epoch,
        "train_kl_per_epoch": train_kl_per_epoch,
        "val_kl_per_epoch": val_kl_per_epoch,
        "train_recon_per_epoch": train_recon_per_epoch,
        "val_recon_per_epoch": val_recon_per_epoch,
        "best_val_rmse": best_val_rmse,
        "best_epoch": best_epoch,
        "model_path": model_filename
    }

def train_autoencoder(model, train_data, val_data, epochs=500, lr=0.001, device='cuda'):
    """
    Training loop for a non-variational autoencoder using reconstruction loss (MSE).
    """
    torch.backends.cudnn.benchmark = True
    
    # Extract model parameters for naming the saved file
    latent_dim = model.latent_dims if hasattr(model, 'latent_dims') else "unknown"
    dropout = model.dropout if hasattr(model, 'dropout') else "unknown"
    
    # Create a unique model filename
    model_filename = f"best_ae_model_ld{latent_dim}_dr{dropout}.pth"

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5)
    
    scaler = torch.amp.GradScaler(device=device)  # For mixed precision training

    train_rmse_per_epoch = []
    val_rmse_per_epoch = []
    train_recon_loss_per_epoch = []
    val_recon_loss_per_epoch = []
    
    best_val_rmse = float('inf')  # Track the best (lowest) validation RMSE
    best_epoch = 0  # Track which epoch had the best performance
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        running_rmse = 0.0
        running_recon_loss = 0.0
        items = 0
        
        for x, _ in train_data:
            batch_size = x.size(0)
            data = x.to(device, non_blocking=True)
            
            opt.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type="cuda"):
                # Forward pass
                x_hat = model(data)
                
                recon_loss = F.mse_loss(data, x_hat, reduction="sum")
                loss = recon_loss
                
                batch_rmse = torch.sqrt(F.mse_loss(data, x_hat, reduction="mean"))
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            items += batch_size
            running_loss += loss.item()
            running_rmse += batch_rmse.item() * batch_size  # weighted sum
            running_recon_loss += recon_loss.item()
        
        scheduler.step(running_loss / items)
        avg_train_rmse = running_rmse / items
        avg_train_recon_loss = running_recon_loss / items
        
        train_rmse_per_epoch.append(avg_train_rmse)
        train_recon_loss_per_epoch.append(avg_train_recon_loss)
        
        # Validation
        model.eval()
        val_rmse = 0.0
        val_recon_loss = 0.0
        val_items = 0
        
        with torch.no_grad():
            for x, _ in val_data:
                batch_size = x.size(0)
                data = x.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type="cuda"):
                    # Forward pass
                    x_hat = model(data)
                    
                    loss = F.mse_loss(data, x_hat, reduction="sum")
                    batch_val_rmse = torch.sqrt(F.mse_loss(data, x_hat, reduction="mean"))
                
                val_items += batch_size
                val_recon_loss += loss.item()
                val_rmse += batch_val_rmse.item() * batch_size
        
        avg_val_rmse = val_rmse / val_items
        avg_val_recon_loss = val_recon_loss / val_items
        
        val_rmse_per_epoch.append(avg_val_rmse)
        val_recon_loss_per_epoch.append(avg_val_recon_loss)
        
        # Check and save the best model state if current validation RMSE is lower
        if avg_val_rmse < best_val_rmse:
            print(f"Epoch {epoch+1}: Saving best model state with RMSE: {avg_val_rmse:.4f}")
            best_val_rmse = avg_val_rmse
            best_epoch = epoch + 1
            
            # Save the best model weights to disk
            torch.save(model.state_dict(), model_filename)
            print(f"Best model saved to: {model_filename}")
        
        print(f"Epoch {epoch+1}, Train RMSE: {avg_train_rmse:.4f}, Val RMSE: {avg_val_rmse:.4f}, " +
              f"Recon Loss (Train): {avg_train_recon_loss:.4f}, Recon Loss (Val): {avg_val_recon_loss:.4f}")
        
    print(f"Training complete. Best model was from epoch {best_epoch} with validation RMSE: {best_val_rmse:.4f}")
    print(f"Best model saved to: {model_filename}")
    
    return {
        "train_rmse_per_epoch": train_rmse_per_epoch,
        "val_rmse_per_epoch": val_rmse_per_epoch,
        "train_recon_loss_per_epoch": train_recon_loss_per_epoch,
        "val_recon_loss_per_epoch": val_recon_loss_per_epoch,
        "best_val_rmse": best_val_rmse,
        "best_epoch": best_epoch,
        "model_path": model_filename
    }

def kl_divergence_loss(mean, logvar):
    """
    Compute KL divergence loss for VAE
    """
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return kl_loss


def vae_loss(x, x_hat, mean, logvar, kl_weight=1.0, reduction="sum"):
    """
    Combined VAE loss: reconstruction + KL divergence
    """
    if reduction == "sum":
        recon_loss = F.mse_loss(x, x_hat, reduction="sum")
    else:
        recon_loss = F.mse_loss(x, x_hat, reduction="mean")

    kl_loss = kl_divergence_loss(mean, logvar)

    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss

def select_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print("Using device:", device)
    print()

    if device.type == 'cuda':
        print("CUDA Device Name:", torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("  Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("  Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
    elif device.type == 'mps':
        print("Using MPS backend on macOS. (Detailed memory info may not be available.)")

    return device

def prep_fa_flattned_data(dataset, batch_size=64):
    """
    Prepares PyTorch dataloaders for training, testing, and validation.
    These dataloaders select the fa tracts ONLY and flatten them to "create" more data.

    Parameters
    ----------
    dataset : AFQDataset
        The dataset to extract fa tracts from and flatten
    batch_size : int
        The batch size to be used.

    Returns
    -------
    tuple:
        The FA dataset,
        New Training data loader,
        New Test data loader,
        New Validation data loader.
    """
    torch_dataset_fa, train_loader_fa, test_loader_fa, val_loader_fa = prep_fa_dataset(
        dataset, batch_size=batch_size
    )

    class AllTractsDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset):
            self.original_dataset = original_dataset
            self.sample_count = len(original_dataset)
            self.tract_count = original_dataset[0][0].shape[0]

        def __len__(self):
            return self.sample_count * self.tract_count

        def __getitem__(self, idx):
            sample_idx = idx // self.tract_count
            tract_idx = idx % self.tract_count

            x, y = self.original_dataset[sample_idx]

            tract_data = x[tract_idx : tract_idx + 1, :].clone()

            return tract_data, y

    all_tracts_train_dataset = AllTractsDataset(train_loader_fa.dataset)
    all_tracts_test_dataset = AllTractsDataset(test_loader_fa.dataset)
    all_tracts_val_dataset = AllTractsDataset(val_loader_fa.dataset)

    all_tracts_train_loader = torch.utils.data.DataLoader(
        all_tracts_train_dataset, batch_size=batch_size, shuffle=True
    )
    all_tracts_test_loader = torch.utils.data.DataLoader(
        all_tracts_test_dataset, batch_size=batch_size, shuffle=False
    )
    all_tracts_val_loader = torch.utils.data.DataLoader(
        all_tracts_val_dataset, batch_size=batch_size, shuffle=False
    )

    return (
        torch_dataset_fa,
        all_tracts_train_loader,
        all_tracts_test_loader,
        all_tracts_val_loader,
    )

def prep_fa_dataset(dataset, target_labels="dki_fa", batch_size=32):
    """
    Extracts features that match the specified label from the provided dataset and
    prepares the dataset for training.
    """
    # Can be single target or a list of targets
    if isinstance(target_labels, str):
        features = [target_labels]
    else:
        features = target_labels

    fa_indices = []
    for i, fname in enumerate(dataset.feature_names):
        if any(feature in fname for feature in features):
            fa_indices.append(i)

    if not fa_indices:
        available_features = sorted(
            {fname.split("_")[0] for fname in dataset.feature_names}
        )
        raise ValueError(
            f"No features found matching patterns: {features}. "
            f"Features found: {available_features}"
        )

    X_fa = dataset.X[:, fa_indices]
    feature_names_fa = [dataset.feature_names[i] for i in fa_indices]
    dataset_fa = AFQDataset(
        X=X_fa,
        y=dataset.y,
        groups=dataset.groups,
        feature_names=feature_names_fa,
        group_names=dataset.group_names,
        target_cols=dataset.target_cols,
        subjects=dataset.subjects,
        sessions=dataset.sessions,
        classes=dataset.classes,
    )
    return prep_pytorch_data(dataset_fa, batch_size=batch_size)

def prep_first_tract_data(dataset, batch_size=32):
    """
    Prepares PyTorch dataloaders for training, testing, and validation.
    These dataloaders select the first tract ONLY.

    Parameters
    ----------
    dataset : AFQDataset
        The dataset to extract the first tract from.
    batch_size : int
        The batch size to be used.

    Returns
    -------
    tuple:
        PyTorch dataset,
        New Training data loader,
        New Test data loader,
        New Validation data loader.
    """
    torch_dataset, train_loader, test_loader, val_loader = prep_pytorch_data(
        dataset, batch_size=batch_size
    )

    class FirstTractDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset):
            self.original_dataset = original_dataset

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, idx):
            x, y = self.original_dataset[idx]
            tract_data = x[0:1, :].clone()
            return tract_data, y

    first_tract_train_dataset = FirstTractDataset(train_loader.dataset)
    first_tract_test_dataset = FirstTractDataset(test_loader.dataset)
    first_tract_val_dataset = FirstTractDataset(val_loader.dataset)

    # Create first tract data loaders
    first_tract_train_loader = torch.utils.data.DataLoader(
        first_tract_train_dataset, batch_size=batch_size, shuffle=True
    )
    first_tract_test_loader = torch.utils.data.DataLoader(
        first_tract_test_dataset, batch_size=batch_size, shuffle=False
    )
    first_tract_val_loader = torch.utils.data.DataLoader(
        first_tract_val_dataset, batch_size=batch_size, shuffle=False
    )

    return (
        torch_dataset,
        first_tract_train_loader,
        first_tract_test_loader,
        first_tract_val_loader,
    )