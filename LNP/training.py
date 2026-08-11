import torch
import torch.nn as nn
from typing import Optional, Dict, List, Callable
from pathlib import Path
from tqdm import tqdm
import numpy as np

class DelayedReduceLROnPlateau:
    """
    Custom Scheduler: 
    Does absolutely nothing for 'warmup_steps' epochs.
    After warmup, acts as a standard ReduceLROnPlateau, halving the LR if 
    a given metric stops improving for 'patience' epochs.
    """
    def __init__(self, optimizer, warmup_steps, mode='min', factor=0.5, patience=10, min_lr=1e-6):
        self.warmup_steps = warmup_steps
        self.current_epoch = 0
        self.optimizer = optimizer
        
        # Initialize the underlying PyTorch scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=mode, 
            factor=factor, 
            patience=patience, 
            min_lr=min_lr
        )

    def step(self, metric):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_steps:
            # During warmup, we do nothing to the learning rate.
            # We also don't feed the metric to the underlying scheduler, 
            # so early volatile losses don't set an impossibly good baseline.
            pass
        else:
            # After warmup, pass the metric to the plateau scheduler
            self.scheduler.step(metric)
            
    def get_last_lr(self):
        # Helper to easily print the current learning rate
        return [group['lr'] for group in self.optimizer.param_groups]


def train_np(
    train_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int = 10,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    gradient_clip: Optional[float] = None,
    checkpoint_dir: Optional[str] = None,
    early_stopping_patience: Optional[int] = None,
    metric_fn: Optional[Callable] = None,
    verbose: bool = False,
    print_every: int = 1,
    forward_fn: Optional[Callable] = None,
    is_meta_learning: bool = False,
    beta_schedule: Optional[List[float]] = None,
    early_stopping_start_epoch: int = 0,
) -> Dict[str, List[float]]:
    """
    Train a Neural Process or standard model with comprehensive tracking and validation.
    
    Supports both standard supervised learning (x, y) and meta-learning with
    context-target splits (x_context, y_context, x_target, y_target).
    
    Args:
        train_loader: DataLoader for training data.
        model: The model to train.
        optimizer: Optimizer for training.
        loss_fn: Loss function or callable. For meta-learning, should accept
                 (y_pred, y_target, z_mu, z_log_var) and return (loss, recon, kl).
        device: Device to train on (cpu/cuda).
        epochs: Number of training epochs.
        val_loader: Optional validation DataLoader.
        scheduler: Optional learning rate scheduler.
        gradient_clip: Optional gradient clipping value.
        checkpoint_dir: Directory to save model checkpoints.
        early_stopping_patience: Stop if validation loss doesn't improve for N epochs.
        metric_fn: Optional additional metric function(outputs, targets) -> float.
        verbose: Whether to print progress.
        print_every: Print training progress every N epochs. Default: 1 (every epoch).
        forward_fn: Optional custom forward function(model, batch, device) -> (outputs, *extras).
                   If None, uses default forward pass.
        is_meta_learning: If True, expects 4-tuple batches and tracks reconstruction/KL losses.
        beta_schedule: Optional list of beta values, one per epoch. If provided, sets
                       loss_fn.beta before each epoch (KL annealing). Length should equal epochs.
        early_stopping_start_epoch: Do not apply early stopping or update best-model checkpoint
                                    before this epoch (0-indexed). Set to warmup_steps to avoid
                                    premature stopping during KL annealing warmup.
        
    Returns:
        Dictionary with training history: {'train_loss', 'val_loss', 'train_metric', 'val_metric'}
    """
    
    history = {k: [] for k in ['train_loss', 'val_loss', 'train_metric', 'val_metric', 'learning_rates']}
    if is_meta_learning:
        history.update({'train_recon': [], 'train_kl': [], 'train_param': []})
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
    # INIZIALIZZA LO SCALER PER LA PRECISIONE MISTA (AMP)
    # L'autocast accelera sia su CUDA che su Mac (MPS)
    use_autocast = device.type in ['cuda', 'mps']

    # Il GradScaler serve SOLO per CUDA
    use_scaler = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # RISOLVI LE CONDIZIONI FUORI DAL LOOP (Evita di valutarle ad ogni batch)
    has_beta = hasattr(loss_fn, 'beta')
    use_forward_fn = forward_fn is not None

    for epoch in range(epochs):
        if beta_schedule is not None and has_beta:
            loss_fn.beta = beta_schedule[min(epoch, len(beta_schedule) - 1)]

        model.train()
        train_loss, train_metric, train_recon, train_kl, train_param = 0.0, 0.0, 0.0, 0.0, 0.0
        num_batches = 0
        
        iterator = train_loader#tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else train_loader
        
        for batch in iterator:
            # TRASFERIMENTO DATI ASINCRONO E PULITO
            batch = [b.to(device, non_blocking=True) for b in batch]
            optimizer.zero_grad(set_to_none=True) # Leggermente più veloce di zero_grad() standard
            
            # CONTESTO AMP PER IL FORWARD E LA LOSS
            with torch.autocast(device_type=device.type, enabled=use_autocast):
                if is_meta_learning or len(batch) >= 4:
                    x_c, y_c, x_t, y_t = batch[:4]
                    
                    if use_forward_fn:
                        result = forward_fn(model, (x_c, y_c, x_t, y_t), device)
                    else:
                        result = model(x_c, y_c, x_t, y_t)
                    
                    # UNPACKING DINAMICO (Niente if len(result) == ...)
                    y_pred_mu, y_pred_var, *latents = result
                    
                    # Gestione elegante dei latenti
                    # After: y_pred_mu, y_pred_var, *latents = result
                    if len(latents) >= 6:
                        z_c_mu, z_c_var, z_t_mu, z_t_var = latents[:4]
                        param_mu, param_var = latents[4], latents[5]
                    elif len(latents) >= 4:
                        z_c_mu, z_c_var, z_t_mu, z_t_var = latents[:4]
                        param_mu, param_var = None, None
                    else:
                        z_c_mu, z_c_var = latents[:2]
                        z_t_mu, z_t_var = z_c_mu, z_c_var
                        param_mu, param_var = None, None
                    
                    # theta_true comes from a 5th batch element if present
                    theta_true = batch[4].to(device, non_blocking=True) if len(batch) >= 5 else None

                    # Calcolo Loss
                    loss_result = loss_fn(
                        y_pred_mu, y_pred_var, y_t,
                        z_c_mu, z_c_var, z_t_mu, z_t_var,
                        param_mu, param_var, theta_true
                        )                  
                    # In the loss unpacking:
                    if isinstance(loss_result, tuple):
                        loss, recon_loss, kl_loss, param_loss = loss_result
                        train_recon  += recon_loss.item()
                        train_kl     += kl_loss.item()
                        train_param  += param_loss.item()
                    else:
                        loss = loss_result
                else:
                    # STANDARD SUPERVISED
                    x, y = batch[:2]
                    outputs = forward_fn(model, (x, y), device) if use_forward_fn else model(x)
                    loss = loss_fn(outputs, y)
                    if metric_fn:
                        train_metric += metric_fn(outputs, y)

            # BACKWARD PASS TRAMITE SCALER (AMP)
            scaler.scale(loss).backward()
            
            if gradient_clip is not None:
                # Per il clipping con AMP bisogna fare prima l'unscale
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            num_batches += 1
            
            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({'loss': loss.item()})
        
        # Calculate epoch averages
        avg_train_loss = train_loss / num_batches
        avg_train_metric = train_metric / num_batches if metric_fn else 0.0
        history['train_loss'].append(avg_train_loss)
        history['train_metric'].append(avg_train_metric)
        
        if is_meta_learning:
            history['train_recon'].append(train_recon / num_batches)
            history['train_kl'].append(train_kl / num_batches)
            history['train_param'].append(train_param / num_batches)
        
        # Validation phase
        if val_loader is not None:
            # ---------------------------------------------------------
            # NEW: Override beta to target beta (1.0) for validation
            # ---------------------------------------------------------
            current_beta = None
            if has_beta:
                current_beta = loss_fn.beta
                # Automatically extract the target beta (last value in schedule) or default to 1.0
                target_beta = beta_schedule[-1] if beta_schedule is not None else 1.0
                loss_fn.beta = target_beta
            
            val_result = evaluate(
                model, val_loader, loss_fn, device, metric_fn, 
                forward_fn=forward_fn, is_meta_learning=is_meta_learning
            )
            
            # ---------------------------------------------------------
            # NEW: Restore the training beta for the next print/epoch
            # ---------------------------------------------------------
            if has_beta and current_beta is not None:
                loss_fn.beta = current_beta
                
            val_loss, val_metric = val_result
                
            history['val_loss'].append(val_loss)
            history['val_metric'].append(val_metric)
            
            # Early stopping / best-model tracking only after warmup is complete
            if epoch >= early_stopping_start_epoch:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    if checkpoint_dir:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_val_loss,
                        }, Path(checkpoint_dir) / 'best_model.pt')
                else:
                    patience_counter += 1
                    
                # Check early stopping
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, DelayedReduceLROnPlateau) or isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if val_loader else avg_train_loss)
            else:
                scheduler.step()
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        if verbose and ((epoch + 1) % print_every == 0 or epoch == 0 or epoch == epochs - 1):
            msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}"
            if is_meta_learning:
                beta_str = f", β={loss_fn.beta:.4f}" if hasattr(loss_fn, 'beta') else ""
                msg += f" (Recon: {history['train_recon'][-1]:.4f}, KL: {history['train_kl'][-1]:.4f}{beta_str})"
            if metric_fn and not is_meta_learning:
                msg += f", Train Metric: {avg_train_metric:.4f}"
            if val_loader:
                msg += f", Val Loss: {val_loss:.4f}"
                if metric_fn and not is_meta_learning:
                    msg += f", Val Metric: {val_metric:.4f}"
            if scheduler:
                msg += f", LR: {optimizer.param_groups[0]['lr']:.6f}"
            print(msg)
    
    # Save final model
    if checkpoint_dir:
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
        }, Path(checkpoint_dir) / 'final_model.pt')
    
    return history


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    metric_fn: Optional[Callable] = None,
    forward_fn: Optional[Callable] = None,
    is_meta_learning: bool = False,
) -> tuple[float, float]:
    """
    Evaluate model on a dataset using optimized mixed precision and data transfer.
    """
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    num_batches = 0
    
    # Valutazioni fuori dal loop
    use_autocast = device.type in ['cuda', 'mps']
    use_forward_fn = forward_fn is not None

    with torch.no_grad():
        for batch in data_loader:
            # Trasferimento dati asincrono
            batch = [b.to(device, non_blocking=True) for b in batch]
            
            # AMP per l'inferenza (niente scaler, solo autocast)
            with torch.autocast(device_type=device.type, enabled=use_autocast):
                if is_meta_learning or len(batch) >= 4:
                    x_c, y_c, x_t, y_t = batch[:4]
                    
                    if use_forward_fn:
                        result = forward_fn(model, (x_c, y_c, x_t, y_t), device)
                    else:
                        result = model(x_c, y_c, x_t, y_t)
                    
                    # Unpacking dinamico
                    y_pred_mu, y_pred_var, *latents = result
                    
                    if len(latents) >= 4:
                        z_c_mu, z_c_var, z_t_mu, z_t_var = latents[:4]
                    else:
                        z_c_mu, z_c_var = latents[:2]
                        z_t_mu, z_t_var = z_c_mu, z_c_var # Fallback
                    
                    # Calcolo Loss
                    try:
                        loss_result = loss_fn(y_pred_mu, y_pred_var, y_t, z_c_mu, z_c_var, z_t_mu, z_t_var)
                    except TypeError:
                        loss_result = loss_fn(y_pred_mu, y_pred_var, y_t, z_c_mu, z_c_var)
                    
                    # In evaluation ci interessa solo la loss totale, non recon e kl separati
                    loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
                    
                else:
                    # STANDARD SUPERVISED
                    x, y = batch[:2]
                    outputs = forward_fn(model, (x, y), device) if use_forward_fn else model(x)
                    loss = loss_fn(outputs, y)
                    
                    if metric_fn is not None:
                        total_metric += metric_fn(outputs, y)
            
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_metric = total_metric / num_batches if metric_fn else 0.0

    return avg_loss, avg_metric
