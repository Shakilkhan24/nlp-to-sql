import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import os
import wandb

from model import Transformer
from data import WikiSQLDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent

import yaml 
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

training_cfg = config['training']
model_cfg = config['model']
data_cfg = config['data']
paths_cfg = config['paths']

batch_size = training_cfg['batch_size']
epochs = training_cfg['epochs']
base_lr = training_cfg['learning_rate']
grad_clip_norm = training_cfg['grad_clip_norm']
ignore_index = model_cfg['ignore_index']
warmup_steps = training_cfg['warmup_steps']
label_smoothing = training_cfg['label_smoothing']

checkpoint_path = Path(paths_cfg.get("checkpoint", "transformer_model.pt"))
if not checkpoint_path.is_absolute():
    checkpoint_path = BASE_DIR / checkpoint_path

train_data_path = Path(paths_cfg.get("train_data", os.path.join("MyTrials", "train.json")))
if not train_data_path.is_absolute():
    train_data_path = BASE_DIR / train_data_path

if train_data_path.exists():
    print(f"train data path: {train_data_path}")
else:
    raise FileNotFoundError(f"{train_data_path} not found.")

dataset = WikiSQLDataset(
    str(train_data_path),
    tokenizer_name=data_cfg.get("tokenizer_name", "t5-small"),
    max_len=data_cfg.get("max_len", 128),
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
vocab_size = dataset.tokenizer.vocab_size

# Model hyperparameters
d_model = model_cfg['d_model']
num_heads = model_cfg['num_heads']
num_encoder_layers = model_cfg['num_encoder_layers']
num_decoder_layers = model_cfg['num_decoder_layers']
d_ff = model_cfg['d_ff']
max_len = model_cfg['max_len']


# Create new model
print("Creating new model...")
model = Transformer(
    vocab_size=vocab_size, 
    d_model=d_model, 
    num_heads=num_heads,
    num_encoder_layers=num_encoder_layers, 
    num_decoder_layers=num_decoder_layers,
    d_ff=d_ff, 
    max_len=max_len
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

# Use label smoothing for better generalization
criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)

# learning rate scheduler with warmup
def get_lr_multiplier(step):
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_multiplier)

# Initialize wandb
wandb.init(
    project="transformer-sql-fixed",
    config={
        "batch_size": batch_size,
        "epochs": epochs,
        "base_learning_rate": base_lr,
        "grad_clip_norm": grad_clip_norm,
        "warmup_steps": warmup_steps,
        "label_smoothing": label_smoothing,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "d_ff": d_ff,
        "max_len": max_len,
        "vocab_size": vocab_size,
        "dataset_size": len(dataset),
        "device": str(device),
    }
)

# Training Loop
print(f"\nthe device is:  {device}")

print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Training for {epochs} epochs\n")

global_step = 0
best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        src = batch["encoder_input_ids"].to(device)
        tgt_input = batch["decoder_input_ids"].to(device)
        tgt_labels = batch["labels"].to(device)
        
        tgt_labels_flat = tgt_labels.contiguous().view(-1)

        logits = model(src, tgt_input)
        logits = logits.view(-1, vocab_size)

        loss = criterion(logits, tgt_labels_flat)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"WARNING: NaN loss at epoch {epoch+1}, batch {batch_idx+1}. Skipping batch.")
            continue
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        global_step += 1
        
        # log to wandb
        wandb.log({
            "batch_loss": loss.item(),
            "learning_rate": scheduler.get_last_lr()[0],
            "grad_norm": grad_norm.item(),
            "epoch": epoch + 1,
            "global_step": global_step,
        })
        
        if (batch_idx + 1) % 100 == 0:
            avg_loss_so_far = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Avg: {avg_loss_so_far:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"GradNorm: {grad_norm.item():.4f}")
        
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    current_epoch = epoch + 1
    
    wandb.log({
        "epoch_loss": avg_loss,
        "epoch": current_epoch,
    })
    
    print(f"\n{'='*60}")
    print(f"Epoch {current_epoch}/{epochs} | Average Loss: {avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        print(f"best loss found so far.")
    
    print(f"{'='*60}\n")
    
    # Save checkpoint after each epoch
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': current_epoch,
        'global_step': global_step,
        'vocab_size': vocab_size,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_encoder_layers': num_encoder_layers,
        'num_decoder_layers': num_decoder_layers,
        'd_ff': d_ff,
        'max_len': max_len,
        'loss': avg_loss,
        'best_loss': best_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}\n")

print(f"\nTraining completed!")
print(f"Best loss achieved: {best_loss:.4f}")
print(f"Final model saved to {checkpoint_path}")

wandb.finish()