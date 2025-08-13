'''
written by: Catherine Breen
July 1, 2024

Training script for users to fine tune model from Breen et. al 2024
Please cite: 

Breen, C. M., Currier, W. R., Vuyovich, C., Miao, Z., & Prugh, L. R. (2024). 
Snow Depth Extraction From Time‐Lapse Imagery Using a Keypoint Deep Learning Model. 
Water Resources Research, 60(7), e2023WR036682. https://doi.org/10.1029/2023WR036682

example run (after updating config)
python src/train.py --use_lstm
'''

# Import startup libraries
import argparse
import tomli as tomllib
import os
import IPython

# Argument parser
parser = argparse.ArgumentParser(description="Train a model on a set of images")
parser.add_argument(
    "--model",
    required=False,
    help='model to train, default is "models/CO_and_WA_model.pth"',
)
parser.add_argument("--path", help="directory where images are located")
parser.add_argument(
    "--device", required=False, help='device to use for training ("cpu" or "cuda")'
)
parser.add_argument(
    "--output", required=False, help="directory in which to store trained models"
)
parser.add_argument(
    "--epochs", required=False, help="epochs"
)
parser.add_argument(
    "--lr", required=False, help="the learning rate of the model"
)
parser.add_argument(
    "--use_lstm", required=False, help="use LSTM model for sequence training", action="store_true"
)
parser.add_argument(
    "--sequence_length", required=False, help="sequence length for LSTM", type=int
)
parser.add_argument(
    "--progressive_training", required=False, help="use progressive training (LSTM first, then fine-tune)", action="store_true"
)
parser.add_argument(
    "--no_confirm", required=False, help="skip confirmation", action="store_true"
)
args = parser.parse_args()

# Get arguments from config file if they weren't specified
with open("config.toml", "rb") as configfile:
    config = tomllib.load(configfile)
if not args.model:
    args.model = config["paths"]["trainee_model"]
if not args.path:
    args.path = config["paths"]["input_images"]
if not args.device:
    args.device = config["training"]["device"]
if not args.output:
    args.output = config["paths"]["models_output"]
if not args.epochs:
    args.epochs = config["training"]["epochs"]
if not args.lr:
    args.lr = config["training"]["lr"]
if not args.sequence_length:
    args.sequence_length = config.get("training", {}).get("sequence_length", 5)

if not args.use_lstm:  # This will be False if not specified on command line
    args.use_lstm = config["training"].get("use_lstm", False)
if not args.progressive_training:  # This will be False if not specified on command line
    args.progressive_training = config["training"].get("progressive_training", False)

# Confirmation
if not args.no_confirm:
    print(
        "\n\n# The following options were specified in config.toml or as arguments:\n"
    )
    if (args.model.startswith("/")):
        print("Model to train:\n" + str(args.model) + "\n")
    else:
        print("Model to train:\n" + os.getcwd() + "/" + str(args.model) + "\n")
    if (args.path.startswith("/")):
        print("Directory where images are located:\n" + str(args.path) + "\n")
    else:
        print("Directory where images are located:\n" + os.getcwd() + "/" + str(args.path) + "\n")
    
    print("Device to use:\n" + args.device + "\n")
    print("LR:\n" + str(args.lr) + "\n")
    print("Epochs:\n" + str(args.epochs) + "\n")
    
    if args.use_lstm:
        print("Using LSTM model with sequence length:\n" + str(args.sequence_length) + "\n")
        print("Progressive training:\n" + str(args.progressive_training) + "\n")
    else:
        print("Using standard CNN model\n")
        
    confirmation = str(input("\nIs this OK? (y/n) "))
    if confirmation.lower() != "y":
        if confirmation.lower() == "n":
            print(
                "\nEdit the config file, located at",
                os.getcwd()
                + "/config.toml, to your liking, or edit the command line arguments if they were specified, and then re-run this file.\n",
            )
        else:
            print("Invalid input.\n")
        quit()

# Import all libraries
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
import utils
from model import snowPoleResNet50, snowPoleResNet50WithLSTM, ProgressiveFineTuning
from tqdm import tqdm
import IPython
import numpy as np
from pathlib import Path
from model_download import download_models

# Import dataset - you'll need to modify this for sequence data
if args.use_lstm:
    from dataset import train_data_sequence, train_loader_sequence, valid_data_sequence, valid_loader_sequence
    train_data, train_loader = train_data_sequence, train_loader_sequence
    valid_data, valid_loader = valid_data_sequence, valid_loader_sequence
else:
    from dataset import train_data, train_loader, valid_data, valid_loader

matplotlib.style.use('ggplot')

## create output path
if not os.path.exists(f"{args.output}"):
    os.makedirs(f"{args.output}", exist_ok=True)

# Initialize model based on type
if args.use_lstm:
    model = snowPoleResNet50WithLSTM(
        pretrained=True, 
        requires_grad=False,  # Start with frozen CNN
        pretrained_cnn_path=args.model,
        sequence_length=args.sequence_length
    ).to(args.device)
    
    # Setup progressive training if requested
    if args.progressive_training:
        trainer = ProgressiveFineTuning(model)
        trainer.stage1_train_lstm_only()
        optimizer = trainer.get_optimizer(stage=1, lstm_lr=args.lr)
        print("Starting with Stage 1: LSTM training only")
    else:
        # Train everything from the start
        model.unfreeze_cnn('all')
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    # Original CNN model
    model = snowPoleResNet50(pretrained=True, requires_grad=False).to(args.device)
    checkpoint = torch.load(args.model, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint["model_state_dict"])
    print("fine-tuned model loaded...")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

criterion = nn.SmoothL1Loss()

# Modified training function
def fit(model, dataloader, data):
    print('Training')
    model.to(args.device)
    model.train()
    train_running_loss = 0.0
    counter = 0
    num_batches = int(len(data)/dataloader.batch_size)
    IPython.embed()
    #dataloader.dataset[0]  # First sample
    
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1

        if args.use_lstm:
            # For LSTM: data contains sequences
            image, keypoints = data["sequence"].to(args.device), data["keypoints"].to(args.device)
        else:
            # For CNN: data contains single images
            image, keypoints = data["image"].to(args.device), data["keypoints"].to(args.device)

        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, keypoints)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss/counter
    return train_loss

# Modified validation function
def validate(model, dataloader, data, epoch):
    print("Validating")
    model.to(args.device)
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    num_batches = int(len(data)/dataloader.batch_size)
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1

            if args.use_lstm:
                image, keypoints = data["sequence"].to(args.device), data["keypoints"].to(args.device)
            else:
                image, keypoints = data["image"].to(args.device), data["keypoints"].to(args.device)

            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            valid_running_loss += loss.item()
            
            # Plot validation results
            if not os.path.exists(args.output):
                os.makedirs(args.output, exist_ok=True)
            if (epoch + 1) % 1 == 0 or i == 20:
                if args.use_lstm:
                    # For LSTM, plot the last frame of the sequence
                    utils.valid_keypoints_plot(image[:, -1, :, :, :], outputs, keypoints, epoch)
                else:
                    utils.valid_keypoints_plot(image, outputs, keypoints, epoch)
        
    valid_loss = valid_running_loss/counter
    return valid_loss

# Training loop with progressive training support
train_loss = []
val_loss = []
best_loss_val = np.inf
best_loss_val_epoch = 0
stage_switch_epoch = args.epochs // 2 if args.progressive_training else -1

for epoch in range(args.epochs):
    print(f"Epoch {epoch+1} of {args.epochs}")
    
    # Switch to stage 2 for progressive training
    if args.use_lstm and args.progressive_training and epoch == stage_switch_epoch:
        print("Switching to Stage 2: Fine-tuning all layers")
        trainer.stage2_finetune_all(unfreeze_layers='last_block')
        optimizer = trainer.get_optimizer(stage=2, lstm_lr=args.lr, cnn_lr=args.lr/10)
    
    train_epoch_loss = fit(model, train_loader, train_data)
    val_epoch_loss = validate(model, valid_loader, valid_data, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {val_epoch_loss:.4f}')
    
    # Save model every 50 epochs
    if (epoch % 50) == 0:
        torch.save(
            {
                "epoch": args.epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": criterion,
                "model_type": "lstm" if args.use_lstm else "cnn",
                "sequence_length": args.sequence_length if args.use_lstm else None,
            },
            f"{args.output}/model_epoch{epoch}.pth",
        )

    # Early stopping
    if val_epoch_loss < best_loss_val:
        best_loss_val = val_epoch_loss
        best_loss_val_epoch = epoch
    elif epoch > best_loss_val_epoch + 10:
        break

# Save final results
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{args.output}/loss.png")
plt.close()

torch.save(
    {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": criterion,
        "model_type": "lstm" if args.use_lstm else "cnn",
        "sequence_length": args.sequence_length if args.use_lstm else None,
    },
    f"{args.output}/model.pth",
)

print("DONE TRAINING")