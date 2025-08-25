#current model structure
#Input Images → CNN (ResNet50) → Feature Maps → LSTM → FC Layer → Output Keypoints

import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels  

class snowPoleResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad, hidden_size, num_layers, num_classes):
        super(snowPoleResNet50, self).__init__()
        
        # Load pretrained ResNet50
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
        
        # Remove the original classifier
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        
        # Add LSTM layers
        # ResNet50 outputs 2048 features, so input_size should be 2048
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True)
        
        # Final output layer for keypoints
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, channels, height, width)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Reshape to process all frames at once
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        
        # Extract features using ResNet50
        features = self.feature_extractor(x)  # (batch_size*seq_len, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size*seq_len, 2048)
        
        # Reshape back to sequence format
        features = features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 2048)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(features)  # (batch_size, seq_len, hidden_size)
        
        # Use the last time step output
        final_output = self.fc(lstm_out[:, -1, :])  # (batch_size, num_classes)
        
        return final_output