"""
written by: Catherine Breen
July 1, 2024

Training script for users to fine tune model from Breen et. al 2024
Please cite: 

Breen, C. M., Currier, W. R., Vuyovich, C., Miao, Z., & Prugh, L. R. (2024). 
Snow Depth Extraction From Time‐Lapse Imagery Using a Keypoint Deep Learning Model. 
Water Resources Research, 60(7), e2023WR036682. https://doi.org/10.1029/2023WR036682

example run (after updating config)
python src/train.py

"""

# import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
import config
import utils
from model import snowPoleResNet50
from dataset import train_data, train_loader, valid_data, valid_loader
from tqdm import tqdm
import IPython
import os
import numpy as np
from pathlib import Path
from model_download import download_models

matplotlib.style.use("ggplot")
# start_time = time.time()

# Comment out this line to disable dark mode
plt.style.use("./themes/dark.mplstyle")

## create output path
if not os.path.exists(f"{config.OUTPUT_PATH}"):
    os.makedirs(f"{config.OUTPUT_PATH}", exist_ok=True)

# model
model = snowPoleResNet50(pretrained=True, requires_grad=True).to(config.DEVICE)
download_models()

if config.FINETUNE == True:
    model_path = "models/CO_and_WA_model.pth"
    checkpoint = torch.load(model_path, map_location=torch.device(config.DEVICE))
    model.load_state_dict(checkpoint["model_state_dict"])
    print("fine-tuned model loaded...")

else:
    print("this run is not using the pre-trained model!")

# optimizer
optimizer = optim.Adam(model.parameters(), lr=config.LR)
criterion = nn.SmoothL1Loss()


# training function
def fit(model, dataloader, data):
    print("Training")
    model.to(config.DEVICE)  ##
    model.train()
    train_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data) / dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        image, keypoints = data["image"].to(config.DEVICE), data["keypoints"].to(
            config.DEVICE
        )
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, keypoints)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / counter
    return train_loss


# validation function
def validate(model, dataloader, data, epoch):
    print("Validating")
    model.to(config.DEVICE)
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data) / dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            image, keypoints = data["image"].to(config.DEVICE), data["keypoints"].to(
                config.DEVICE
            )
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(
                outputs, keypoints
            )  ## cross entropy loss between input and output
            valid_running_loss += loss.item()
            # plot the predicted validation keypoints after every...
            # ... predefined number of epochs
            if not os.path.exists(config.OUTPUT_PATH):
                os.makedirs(config.OUTPUT_PATH, exist_ok=True)
            if (
                epoch + 1
            ) % 1 == 0 and i == 20:  # make this not 0 to get a different image
                utils.valid_keypoints_plot(image, outputs, keypoints, epoch)

    valid_loss = valid_running_loss / counter
    return valid_loss


train_loss = []
val_loss = []
## early stopping ##
#######################
best_loss_val = np.inf
best_loss_val_epoch = 0
#######################
for epoch in range(config.EPOCHS):

    print(f"Epoch {epoch+1} of {config.EPOCHS}")
    train_epoch_loss = fit(model, train_loader, train_data)
    val_epoch_loss = validate(model, valid_loader, valid_data, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")
    ####### saving model every 50 epochs
    if (epoch % 50) == 0:
        torch.save(
            {
                "epoch": config.EPOCHS,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": criterion,
            },
            f"{config.OUTPUT_PATH}/model_epoch{epoch}.pth",
        )

    ####### early stopping #########
    if val_epoch_loss < best_loss_val:
        best_loss_val = val_epoch_loss
        best_loss_val_epoch = epoch
    elif epoch > best_loss_val_epoch + 10:
        break

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color="orange", label="train loss")
plt.plot(val_loss, color="red", label="validataion loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"{config.OUTPUT_PATH}/loss.png")
plt.close()  # changed from plt.show()
torch.save(
    {
        "epoch": config.EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": criterion,
    },
    f"{config.OUTPUT_PATH}/model.pth",
)  ### the last model
print("DONE TRAINING")
# print("My program took", time.time() - start_time, "to run")
