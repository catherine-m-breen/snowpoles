'''
written by: Catherine Breen
July 1, 2024

Training script for users to fine tune model from Breen et. al 2024
Please cite: 

Breen, C. M., Currier, W. R., Vuyovich, C., Miao, Z., & Prugh, L. R. (2024). 
Snow Depth Extraction From Time‐Lapse Imagery Using a Keypoint Deep Learning Model. 
Water Resources Research, 60(7), e2023WR036682. https://doi.org/10.1029/2023WR036682

example run (after updating config)
python src/train.py

tensorboard --logdir=runs

'''

# Import startup libraries
import argparse
import tomli as tomllib
import os

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

# Confirmation
if not args.no_confirm:
    print(
        "\n\n# The following options were specified in config.toml or as arguments:\n"
    )
    if (args.model.startswith("/")):
        print(
            "Model to train:\n"
            + str(args.model)
            + "\n"
        )
    else:
        print(
            "Model to train:\n"
            + os.getcwd()
            + "/"
            + str(args.model)
            + "\n"
        )
    if (args.path.startswith("/")):
        print(
            "Directory where images are located:\n"
            + str(args.path)
            + "\n"
        )
    else:
        print(
            "Directory where images are located:\n"
            + os.getcwd()
            + "/"
            + str(args.path)
            + "\n"
        )
    print("Device to use:\n" + args.device + "\n")
    if (args.output.startswith("/")):
        print(
            "Directory where generated models will be stored:\n"
            + str(args.output)
            + "\n"
        )
    else:
        print(
            "Directory where generated models will be stored:\n"
            + os.getcwd()
            + "/"
            + str(args.output)
            + "\n"
        )
    print("LR:\n" + str(args.lr) + "\n")
    print("Epochs:\n" + str(args.epochs) + "\n")
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
from model import snowPoleResNet50
from tqdm import tqdm
import IPython
import numpy as np
from pathlib import Path
from model_download import download_models
from dataset import train_data, train_loader, valid_data, valid_loader

matplotlib.style.use('ggplot')
# start_time = time.time() 

# training viz 
from torch.utils.tensorboard import SummaryWriter  # For PyTorch

exp_name = config["paths"]["models_output"].split('/')[-1]
writer = SummaryWriter(f'runs/{exp_name}')


## create output path
if not os.path.exists(f"{args.output}"):
    os.makedirs(f"{args.output}", exist_ok=True)


# model
#model = snowPoleResNet50(pretrained=True, requires_grad=False).to(args.device)
#torch.serialization.add_safe_globals([torch.nn.modules.loss.SmoothL1Loss])
checkpoint = torch.load(args.model, map_location=torch.device(args.device))
# model.load_state_dict(checkpoint["model_state_dict"])

# print("fine-tuned model loaded...")

#############
# Create new model with camera embeddings
num_cameras = 15 ## update with wherever the mapping is ... 
model = snowPoleResNet50(
    pretrained=False,  # Don't load ImageNet weights since we're loading our own
    requires_grad=True,
    num_cameras=num_cameras,
    embedding_dim=64).to(args.device)

# Get the state dict from checkpoint
if 'model_state_dict' in checkpoint:
    pretrained_state_dict = checkpoint['model_state_dict']
else:
    pretrained_state_dict = checkpoint

# Get the new model's state dict
new_state_dict = model.state_dict()

# Load compatible weights from pretrained model
loaded_keys = []
for key, value in pretrained_state_dict.items():
    if key in new_state_dict and new_state_dict[key].shape == value.shape:
        new_state_dict[key] = value
        loaded_keys.append(key)
    else:
        print(f"Skipping key '{key}' - not compatible or not found in new model")

print(f"Successfully loaded {len(loaded_keys)} weight tensors from pretrained model")
print("New components (camera embedding, fusion layers) will be randomly initialized")

# Load the modified state dict
model.load_state_dict(new_state_dict)

#############

# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.SmoothL1Loss()

# training function
def fit(model, dataloader, data):

    # print("Training")
    # model.to(args.device)  ##

    #.embed()
    print('Training')
    model.to(config['training']['device']) ##

    model.train()
    train_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1

        image, keypoints, camera_ids = data["image"].to(args.device), data["keypoints"].to(
            args.device), data['camera_id'].to(args.device)

        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = model(image, camera_ids)
        loss = criterion(outputs, keypoints)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss/counter
    return train_loss


# validation function
def validate(model, dataloader, data, epoch):
    print("Validating")
    model.to(args.device)

    model.eval()
    valid_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1

            image, keypoints, camera_ids = data["image"].to(args.device), data["keypoints"].to(
            args.device), data['camera_id'].to(args.device)

            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image, camera_ids)
            loss = criterion(outputs, keypoints) ## cross entropy loss between input and output
            valid_running_loss += loss.item()
            # plot the predicted validation keypoints after every...
            # ... predefined number of epochs
            if not os.path.exists(args.output):
                os.makedirs(args.output, exist_ok=True)
            if (
                epoch + 1
            ) % 1 == 0 or i == 20:  # make this not 0 to get a different image
                utils.valid_keypoints_plot(image, outputs, keypoints, epoch)
        
    valid_loss = valid_running_loss/counter
    return valid_loss

train_loss = []
val_loss = []
## early stopping ##
#######################
best_loss_val = np.inf
best_loss_val_epoch = 0 
best_model = model
#######################
for epoch in range(args.epochs):

    print(f"Epoch {epoch+1} of {args.epochs}")
    train_epoch_loss = fit(model, train_loader, train_data)
    val_epoch_loss = validate(model, valid_loader, valid_data, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {val_epoch_loss:.4f}')
    ####### saving model every 50 epochs
    if (epoch % 50) == 0:
        torch.save(
            {
                "epoch": args.epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": criterion,
            },
            f"{args.output}/model_epoch{epoch}.pth",
        )

    writer.add_scalar('Loss/train',train_epoch_loss, epoch)
    writer.add_scalar('Loss/validation',val_epoch_loss, epoch)
    writer.flush()

    ####### early stopping #########
    if val_epoch_loss < best_loss_val:
                best_loss_val = val_epoch_loss
                best_loss_val_epoch = epoch
                best_model = epoch
    elif epoch > best_loss_val_epoch + 10:
            epoch = best_model ### save model at lowest val error, rather than 10 epochs later 
            break

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# Add white background and light grey grid
plt.gca().set_facecolor('white')
plt.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)
plt.gca().set_axisbelow(True)  # Put grid lines behind the plot lines
##

plt.savefig(f"{args.output}/loss.png")
plt.close()  # changed from plt.show()
torch.save(
    {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": criterion,
    },
    f"{args.output}/model.pth",
)  ### the last model
writer.close()
print("DONE TRAINING")

# print("My program took", time.time() - start_time, "to run")
