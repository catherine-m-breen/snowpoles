## to launch with tensorboard: 
# tensorboard --logdir=runs

# Import startup libraries
import argparse
import tomli as tomllib
import os

# Import all libraries
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
import utils
from model import snowPoleResNet50WithCamera
from tqdm import tqdm
import IPython
import numpy as np
from pathlib import Path
from model_download import download_models
from dataset import train_data, train_loader, valid_data, valid_loader
import torch.optim.lr_scheduler as lr_scheduler

# training viz 
from torch.utils.tensorboard import SummaryWriter  # For PyTorch

writer = SummaryWriter('runs/warm_up_exp')

matplotlib.style.use('ggplot')
# start_time = time.time() 

from sklearn.neighbors import KernelDensity


# warm up period 
# real-world example that shows it didn't get stuck in local minima: https://link.springer.com/article/10.1007/s40808-024-02211-z 
# other one; Kalra, D. S., & Barkeshli, M. (n.d.). Why Warmup the Learning Rate? Underlying Mechanisms and Improvements.

class WarmupScheduler:
    """
    Scheduler that handles warm-up phase followed by main scheduler
    """
    def __init__(self, optimizer, warmup_epochs, main_scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.main_scheduler = main_scheduler
        self.current_epoch = 0
        
        # Store initial learning rates
        self.initial_lrs = []
        for group in optimizer.param_groups:
            self.initial_lrs.append(group['lr'])
    
    def step(self, val_loss=None):
        if self.current_epoch < self.warmup_epochs:
            # Warm-up phase - linearly increase LR
            warmup_factor = float(self.current_epoch + 1) / float(self.warmup_epochs)
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.initial_lrs[i] * warmup_factor
                print(f"Warm-up: LR set to {group['lr']:.6f} (group {i})")
        else:
            # Main scheduling phase - use ReduceLROnPlateau
            self.main_scheduler.step(val_loss)
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]





def calculate_density_weights(keypoints_array, bandwidth=10.0):
    """
    Calculate weights based on local density - rare positions get higher weights
    """
    # Fit kernel density estimator
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    keypoints_flattened = keypoints_array.reshape(keypoints_array.shape[0], -1)
    kde.fit(keypoints_flattened)
    
    # Get log density for each point
    log_density = kde.score_samples(keypoints_flattened)
    density = np.exp(log_density)
    
    # Invert density to get weights (low density = high weight)
    weights = 1.0 / (density + 1e-8)
    
    # Normalize weights
    weights = weights / np.mean(weights)
    
    return weights


# training function
def fit(model, dataloader, data, config):
    print('Training')
    #dataloader.dataset[0]  # First sample
    model.to(config['training']['device'])  # Use args.device consistently
    model.train()
    train_running_loss = 0.0
    counter = 0
    num_batches = int(len(data)/dataloader.batch_size)
    
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        image, keypoints, camera_ids = data["image"].to(config['training']['device']), data["keypoints"].to(config['training']['device']),
        data["camera_id"].to(config['training']['device'])

        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = model(image, camera_ids)
        # regular loss 
        loss = criterion(outputs, keypoints)
        # weighted loss: 
        batch_indices = data['index']  # You'll need to add this to your dataset
        batch_weights = torch.tensor([sample_weights[idx] for idx in batch_indices])
        loss = (loss * batch_weights.to(config['training']['device'])).mean()
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss/counter
    return train_loss

# validation function
def validate(model, dataloader, data, epoch, config):
    print("Validating")
    model.to(config['training']['device'])
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            image, keypoints, camera_ids = data["image"].to(config['training']['device']), data["keypoints"].to(config['training']['device']),
            data["camera_id"].to(config['training']['device'])

            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image, camera_ids)

            # Debug first batch of first epoch
            if i == 0 and epoch == 0:
                print(f"\n=== DEBUGGING MODEL PREDICTIONS ===")
                print(f"Batch size: {keypoints.shape[0]}")
                print(f"First sample target: {keypoints[0].cpu().numpy()}")
                print(f"First sample prediction: {outputs[0].cpu().numpy()}")
                print(f"All targets - min: {keypoints.min():.3f}, max: {keypoints.max():.3f}")
                print(f"All predictions - min: {outputs.min():.3f}, max: {outputs.max():.3f}")
                print(f"Prediction std: {outputs.std():.6f}")
                print("=====================================\n")
            
            loss = criterion(outputs, keypoints)

            loss = criterion(outputs, keypoints) ## cross entropy loss between input and output
            valid_running_loss += loss.item()
            # plot the predicted validation keypoints after every...
            # ... predefined number of epochs
            if not os.path.exists(config['paths']['models_output']):
                os.makedirs(config['paths']['models_output'], exist_ok=True)
            if (
                epoch + 1
            ) % 1 == 0 or i == 20:  # make this not 0 to get a different image
                utils.valid_keypoints_plot(image, outputs, keypoints, epoch)
        
    valid_loss = valid_running_loss/counter
        # In your validate function, you could log sample images periodically:
    if epoch % 10 == 0:  # Every 10 epochs
        #IPython.embed()
        vis_images = image[:4]
        # Reshape to show all frames: (batch*sequence, channels, height, width)
        batch_size, seq_len, channels, height, width = vis_images.shape
        vis_images = vis_images.reshape(batch_size * seq_len, channels, height, width)
    
        writer.add_images('Validation_Images', vis_images, epoch)
    return valid_loss

###### TRAINING SCRIPT ####
## get args from config file 
parser = argparse.ArgumentParser(description='Train deep learning model.')
parser.add_argument('--config', help='Path to config file', default='config_lstm.toml')
args = parser.parse_args()

print(f'Using config "{args.config}"')
with open(args.config, "rb") as configfile:
    config = tomllib.load(configfile)

## create output path
if not os.path.exists(f"{config['paths']['models_output']}"):
    os.makedirs(f"{config['paths']['models_output']}", exist_ok=True)

# model
#model = snowPoleResNet50(pretrained=True, requires_grad=False).to(args.device)
model = snowPoleResNet50WithCamera(
    pretrained=True, 
    requires_grad=True, ## True if you want to fine-tune the CNN too; try both 
    hidden_size=256,  
    num_layers=2,     
   num_classes=4,   ## could adjust and predict more poles in the image i guess? 
   num_cameras = config['model']['num_cameras'],
   camera_embedding_dim =64
).to(config['training']['device'])

checkpoint = torch.load(config['paths']['trainee_model'], map_location=torch.device(config['training']['device']))
pretrained_state_dict = checkpoint["model_state_dict"]

# Get the current model's state dict
model_state_dict = model.state_dict()

# Filter out the CNN backbone weights that match
# pretrained_backbone = {k: v for k, v in pretrained_state_dict.items() 
#                       if k in model_state_dict and k.startswith('model.')}

pretrained_backbone = {k: v for k, v in pretrained_state_dict.items() 
                      if k in model_state_dict and k.startswith('feature_extractor.')}


# Update only the matching backbone weights
model_state_dict.update(pretrained_backbone)
model.load_state_dict(model_state_dict)
print("Pretrained CNN backbone loaded, LSTM layers initialized randomly...")

# optimizer
#optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer = optim.Adam([
    {'params': model.feature_extractor.parameters(), 'lr': config['training']['lr'] * 0.1},  # Lower LR for CNN
    {'params': model.camera_embedding.parameters(), 'lr': config['training']['lr']},  
    {'params': model.lstm.parameters(), 'lr': config['training']['lr']},
    {'params': model.fc.parameters(), 'lr': config['training']['lr']}
])

main_scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',           # Minimize validation loss
    factor=0.5,          # Reduce LR by 50% when plateau
    patience=5,         # Wait 5 epochs before reducing
    verbose=True,        # Print when LR changes
    min_lr=1e-7,         # Don't go below this LR
    threshold=0.001      # Threshold for measuring improvement
)

warmup_epochs = 5
scheduler = WarmupScheduler(optimizer, warmup_epochs, main_scheduler)

criterion = nn.SmoothL1Loss() 
#criterion = nn.MSELoss()

### this is the equation for the smooth l1 loss: 
# SmoothL1Loss(x) = {
#     0.5 * x^2                if |x| < beta
#     |x| - 0.5 * beta         if |x| ≥ beta
# }

### weighted loss ## 
# Calculate weights once before training
training_keypoints = []  # Collect all training keypoints first
for data in train_loader:
    training_keypoints.append(data['keypoints'].numpy())
training_keypoints = np.vstack(training_keypoints)

sample_weights = calculate_density_weights(training_keypoints)


train_loss = []
val_loss = []
## early stopping ##
#######################
best_loss_val = np.inf
best_loss_val_epoch = 0 
#######################
for epoch in range(config['training']['epochs']):
    print(f"Epoch {epoch+1} of {config['training']['epochs']}")
    train_epoch_loss = fit(model, train_loader, train_data, config)
    val_epoch_loss = validate(model, valid_loader, valid_data, epoch, config)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {val_epoch_loss:.4f}')
    
    scheduler.step(val_epoch_loss)

    current_lrs = scheduler.get_last_lr()

    writer.add_scalar('Loss/train',train_epoch_loss, epoch)
    writer.add_scalar('Loss/validation',val_epoch_loss, epoch)
    writer.add_scalar('Learning_Rate/CNN', current_lrs[0], epoch)  # CNN backbone
    writer.add_scalar('Learning_Rate/Camera_Embedding', current_lrs[1], epoch) 
    writer.add_scalar('Learning_Rate/LSTM', current_lrs[2], epoch)  # LSTM
    writer.add_scalar('Learning_Rate/FC', current_lrs[3], epoch)    # Fully connected
    #writer.add_scalar('Learning_Rate', current_lr, epoch)
    writer.flush()
    
    ####### saving model every 50 epochs
    if (epoch % 50) == 0:
        torch.save(
            {
                "epoch": config['training']['epochs'],
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": criterion,
            },
            f"{config['paths']['models_output']}/model_epoch{epoch}.pth",
        )

    ####### early stopping #########
    if val_epoch_loss < best_loss_val:
                best_loss_val = val_epoch_loss
                best_loss_val_epoch = epoch
    elif epoch > best_loss_val_epoch + 20:
            break

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{config['paths']['models_output']}/loss.png")
plt.close()  # changed from plt.show()
torch.save(
    {
        "epoch": config['training']['epochs'],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": criterion,
    },
    f"{config['paths']['models_output']}/model.pth",
)  ### the last model
writer.close()
print("DONE TRAINING")