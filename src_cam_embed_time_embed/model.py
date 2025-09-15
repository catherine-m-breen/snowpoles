import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels  
import torch
import math
from datetime import datetime
import pandas as pd
import IPython

class snowPoleResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad, num_cameras, embedding_dim=64, time_embedding_dim=32):
        super(snowPoleResNet50, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
        
        # Camera embedding (existing)
        self.camera_embedding = nn.Embedding(num_cameras, embedding_dim)
        
        # Time embedding layers
        self.time_embedding_dim = time_embedding_dim
        self.time_projection = nn.Linear(self.time_embedding_dim, time_embedding_dim)
        
        # Modified fusion layer to include time features
        self.fusion_layer = nn.Linear(2048 + embedding_dim + time_embedding_dim, 512)
        
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(512, 4)
        
        # Initialize embeddings
        nn.init.normal_(self.camera_embedding.weight, 0, 0.1)
        nn.init.xavier_uniform_(self.time_projection.weight)

    def parse_datetime_to_tensor(self, datetime_strings):
        """
        Convert datetime strings to Unix timestamps
        datetime_strings: list or array of strings in format "mm/dd/yyyy HH:MM"
        """
        timestamps = []
        for dt_str in datetime_strings:
            try:
                # Parse the datetime string
                #IPython.embed()
                dt = datetime.strptime(dt_str, "%m/%d/%y %H:%M")
                # Convert to Unix timestamp
                timestamp = dt.timestamp()
                timestamps.append(timestamp)
            except ValueError:
                print(f"Warning: Could not parse datetime string: {dt_str}")
                # Use a default timestamp or the current time
                timestamps.append(datetime.now().timestamp())
        
        return torch.tensor(timestamps, dtype=torch.float32)

    def encode_time(self, timestamps):
        """
        Create sinusoidal time embeddings from Unix timestamps
        timestamps: tensor of shape (batch_size,) containing Unix timestamps
        """
        batch_size = timestamps.shape[0]
        device = timestamps.device
        
        # Normalize timestamps to a reasonable range
        # Using min-max normalization within the batch
        if batch_size > 1:
            min_time = timestamps.min()
            max_time = timestamps.max()
            time_range = max_time - min_time
            if time_range > 0:
                normalized_time = (timestamps - min_time) / time_range
            else:
                normalized_time = torch.zeros_like(timestamps)
        else:
            # For single sample, use a fixed normalization
            # You might want to use global min/max from your dataset here
            normalized_time = timestamps / 1e10  # Rough normalization for Unix timestamps
        
        # Create positional encoding features
        # this is the standard equation from "Attention is all you need"
        # converts time to frequencies so it can learn order rather than treat each number independently (also makes them cyclical)
        div_term = torch.exp(torch.arange(0, self.time_embedding_dim, 2).float() * 
                           (-math.log(10000.0) / self.time_embedding_dim)).to(device)
        
        time_embedding = torch.zeros(batch_size, self.time_embedding_dim).to(device)
        
        # Apply sine to even indices
        time_embedding[:, 0::2] = torch.sin(normalized_time.unsqueeze(1) * div_term)
        # Apply cosine to odd indices
        if self.time_embedding_dim > 1:
            time_embedding[:, 1::2] = torch.cos(normalized_time.unsqueeze(1) * div_term)
        
        return time_embedding

    def forward(self, x, camera_ids=None, datetime_strings=None):
        """
        Forward pass with datetime strings
        
        Args:
            x: input images
            camera_ids: camera ID tensors
            datetime_strings: list of datetime strings in format "mm/dd/yyyy HH:MM"
        """
        batch, _, _, _ = x.shape
        
        # Extract image features
        image_features = self.model.features(x)
        image_features = F.adaptive_avg_pool2d(image_features, 1).reshape(batch, -1)
        
        # Camera embeddings
        camera_embeddings = self.camera_embedding(camera_ids)
        
        # Time embeddings from datetime strings
        if datetime_strings is not None:
            # Convert datetime strings to timestamps
            timestamps = self.parse_datetime_to_tensor(datetime_strings).to(x.device)
            time_embeddings = self.encode_time(timestamps)
            time_embeddings = self.time_projection(time_embeddings)
        else:
            # Fallback: zero time embeddings if no datetime info
            time_embeddings = torch.zeros(batch, self.time_embedding_dim).to(x.device)
            time_embeddings = self.time_projection(time_embeddings)
        
        # Combine all features
        combined_features = torch.cat([image_features, camera_embeddings, time_embeddings], dim=1)
        
        x = F.relu(self.fusion_layer(combined_features))
        x = self.dropout(x)
        output = self.output_layer(x)
        
        return output