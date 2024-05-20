import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
import config
#import config_cpu as config
import utils
from model import snowPoleResNet50
from dataset import train_data, train_loader, valid_data, valid_loader
from tqdm import tqdm
import IPython
import os
import numpy as np 

matplotlib.style.use('ggplot')
start_time = time.time() 

## create output path
if not os.path.exists(f"{config.OUTPUT_PATH}"):
    os.makedirs(f"{config.OUTPUT_PATH}", exist_ok=True)

if config.DEVICE != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{config.DEVICE}" but CUDA not available; falling back to CPU...')
else: print(f'THIS EXPERIMENT USES GPUS: {config.DEVICE}')

# model 
model = snowPoleResNet50(pretrained=True, requires_grad=True).to(config.DEVICE)
## model ## load previous model 
if config.FINETUNE == True:
    checkpoint = torch.load(config.FT_PATH + '/model.pth', map_location=torch.device('cpu'))
        # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    print('fine-tuned model loaded...')

# optimizer
optimizer = optim.Adam(model.parameters(), lr=config.LR)
# we need a loss function which is good for regression like SmmothL1Loss ...
# ... or MSELoss
criterion = nn.SmoothL1Loss()

# training function
def fit(model, dataloader, data):
    print('Training')
    model.to(config.DEVICE) ### added on Aug 24
    model.train()
    train_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
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


# validation function
def validate(model, dataloader, data, epoch):
    print('Validating')
    model.to(config.DEVICE)
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints) ## cross entropy loss between input and output
            valid_running_loss += loss.item()
            # plot the predicted validation keypoints after every...
            # ... predefined number of epochs
            if not os.path.exists(config.OUTPUT_PATH):
                os.makedirs(config.OUTPUT_PATH, exist_ok=True)
            if (epoch+1) % 1 == 0 and i == 20:  # make this not 0 to get a different image
                utils.valid_keypoints_plot(image, outputs, keypoints, epoch)
        
    valid_loss = valid_running_loss/counter
    return valid_loss

train_loss = []
val_loss = []
## early stopping ##
#######################
best_loss_val = np.inf
best_loss_val_epoch = 0 # index of the epoch
#######################
for epoch in range(config.EPOCHS):

    print(f"Epoch {epoch+1} of {config.EPOCHS}")
    train_epoch_loss = fit(model, train_loader, train_data)
    val_epoch_loss = validate(model, valid_loader, valid_data, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {val_epoch_loss:.4f}')
    ####### saving model each epoch
    #IPython.embed()
    if (epoch % 50) == 0:
        torch.save({
            'epoch': config.EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, f"{config.OUTPUT_PATH}/model_epoch{epoch}.pth")

    ####### early stopping #########
    #IPython.embed()
    if val_epoch_loss < best_loss_val:
                best_loss_val = val_epoch_loss
                best_loss_val_epoch = epoch
    elif epoch > best_loss_val_epoch + 10:
            break

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{config.OUTPUT_PATH}/loss.png")
plt.close()  # changed from plt.show()
torch.save({
            'epoch': config.EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, f"{config.OUTPUT_PATH}/model.pth") ### the last model
print('DONE TRAINING')
print("My program took", time.time() - start_time, "to run")