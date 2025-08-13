'''
Catherine Breen
cbreen@uw.edu
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels  

class snowPoleResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad):
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
        # change the final layer
        self.l0 = nn.Linear(2048, 4)  #### the second value is the number of points you want to predict

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0

class snowPoleResNet50WithLSTM(nn.Module):
    def __init__(self, pretrained=True, requires_grad=False, pretrained_cnn_path=None, 
                 hidden_size=256, num_layers=1, sequence_length=5):
        super(snowPoleResNet50WithLSTM, self).__init__()
        
        # Load your pre-trained CNN
        self.cnn = snowPoleResNet50(pretrained=pretrained, requires_grad=requires_grad)
        if pretrained_cnn_path:
            # Load the checkpoint
            checkpoint = torch.load(pretrained_cnn_path, map_location=torch.device('cpu'))   
            # Extract the actual model state dict
            if 'model_state_dict' in checkpoint:
                self.cnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                # If it's already just the state dict
                self.cnn.load_state_dict(checkpoint)
            print(f"Loaded pre-trained CNN from {pretrained_cnn_path}")
        
        # Remove the final layer to get features (2048-dim from ResNet50)
        self.cnn.l0 = nn.Identity()
        
        # Store sequence length for reference
        self.sequence_length = sequence_length
        
        # Add LSTM on top
        self.lstm = nn.LSTM(
            input_size=2048,  # ResNet50 feature dimension
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Final prediction layer
        self.final_layer = nn.Linear(hidden_size, 4)  # 4 keypoints (x1,y1,x2,y2)
        
        # Optional: Add some regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, c, h, w = x.shape
        
        # Extract CNN features for each frame in the sequence
        cnn_features = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]  # (batch_size, c, h, w)
            # Use torch.no_grad() only if CNN is frozen during training
            if not any(param.requires_grad for param in self.cnn.parameters()) and self.training:
                with torch.no_grad():
                    features = self.cnn(frame)  # (batch_size, 2048)
            else:
                features = self.cnn(frame)  # (batch_size, 2048)
            cnn_features.append(features)
        
        # Stack for LSTM: (batch_size, seq_len, 2048)
        lstm_input = torch.stack(cnn_features, dim=1)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout and final prediction
        output = self.dropout(last_output)
        keypoints = self.final_layer(output)
        
        return keypoints
    
    def unfreeze_cnn(self, layers_to_unfreeze='all'):
        """Unfreeze CNN layers for fine-tuning"""
        if layers_to_unfreeze == 'all':
            for param in self.cnn.parameters():
                param.requires_grad = True
            print("Unfroze all CNN layers")
        elif layers_to_unfreeze == 'last_block':
            # Unfreeze only the last ResNet block
            for name, param in self.cnn.named_parameters():
                if 'layer4' in name:  # Last ResNet block
                    param.requires_grad = True
            print("Unfroze last CNN block")
        elif layers_to_unfreeze == 'final_layer':
            # Unfreeze only the final linear layer
            for param in self.cnn.l0.parameters():
                param.requires_grad = True
            print("Unfroze CNN final layer only")

class ProgressiveFineTuning:
    """Helper class for progressive fine-tuning of the LSTM model"""
    def __init__(self, model):
        self.model = model
        
    def stage1_train_lstm_only(self):
        """Stage 1: Train only LSTM, freeze CNN"""
        for param in self.model.cnn.parameters():
            param.requires_grad = False
        for param in self.model.lstm.parameters():
            param.requires_grad = True
        for param in self.model.final_layer.parameters():
            param.requires_grad = True
        print("Stage 1: Training LSTM only, CNN frozen")
        
    def stage2_finetune_all(self, unfreeze_layers='last_block'):
        """Stage 2: Fine-tune entire network"""
        # Unfreeze CNN layers
        self.model.unfreeze_cnn(unfreeze_layers)
        print(f"Stage 2: Fine-tuning with CNN layers: {unfreeze_layers}")
        
    def get_optimizer(self, stage=1, lstm_lr=1e-3, cnn_lr=1e-5):
        """Get optimizer with appropriate learning rates for each stage"""
        if stage == 1:
            # Only LSTM and final layer parameters
            return torch.optim.Adam([
                {'params': self.model.lstm.parameters(), 'lr': lstm_lr},
                {'params': self.model.final_layer.parameters(), 'lr': lstm_lr}
            ])
        else:
            # All trainable parameters with different learning rates
            cnn_params = [p for p in self.model.cnn.parameters() if p.requires_grad]
            lstm_params = list(self.model.lstm.parameters())
            final_params = list(self.model.final_layer.parameters())
            
            param_groups = []
            if cnn_params:
                param_groups.append({'params': cnn_params, 'lr': cnn_lr})
            param_groups.extend([
                {'params': lstm_params, 'lr': lstm_lr},
                {'params': final_params, 'lr': lstm_lr}
            ])
            
            return torch.optim.Adam(param_groups)