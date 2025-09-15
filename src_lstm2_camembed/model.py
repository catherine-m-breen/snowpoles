import torch
import torch.nn as nn
import pretrainedmodels

class snowPoleResNet50WithCamera(nn.Module):
    def __init__(self, pretrained, requires_grad, hidden_size, num_layers, num_classes, num_cameras, camera_embedding_dim=64):
        super(snowPoleResNet50WithCamera, self).__init__()
        
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
        
        # Camera embedding layer
        self.camera_embedding = nn.Embedding(num_cameras, camera_embedding_dim)
        
        # Add LSTM layers with concatenated features
        # ResNet50 outputs 2048 features + camera_embedding_dim
        self.lstm = nn.LSTM(input_size=2048 + camera_embedding_dim, 
                           hidden_size=hidden_size, 
                           num_layers=num_layers, 
                           batch_first=True)
        
        # Final output layer for keypoints
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, camera_ids):
        # x shape: (batch_size, seq_len, channels, height, width)
        # camera_ids shape: (batch_size, seq_len)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Reshape to process all frames at once
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        
        # Extract features using ResNet50
        features = self.feature_extractor(x)  # (batch_size*seq_len, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size*seq_len, 2048)
        
        # Reshape back to sequence format
        features = features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 2048)
        
        # Get camera embeddings and reshape
        #camera_ids_flat = camera_ids.view(-1)  # (batch_size * seq_len)
        #cam_embeddings = self.camera_embedding(camera_ids_flat)  # (batch_size * seq_len, embedding_dim)
        #cam_embeddings = cam_embeddings.view(batch_size, seq_len, -1)  # (batch_size, seq_len, embedding_dim)
        cam_embeddings = self.camera_embedding(camera_ids)  # (batch_size, embedding_dim)
        cam_embeddings = cam_embeddings.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, embedding_dim)
    
        
        # Concatenate image features and camera embeddings
        combined_features = torch.cat([features, cam_embeddings], dim=2)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(combined_features)
        
        # Use the last time step output
        final_output = self.fc(lstm_out[:, -1, :])
        
        return final_output