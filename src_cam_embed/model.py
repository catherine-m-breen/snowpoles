'''
Catherine Breen
cbreen@uw.edu
adapted from: 
https://debuggercafe.com/advanced-facial-keypoint-detection-with-pytorch/

'''

import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels  
import torch

class snowPoleResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad, num_cameras, embedding_dim=64):
    #def __init__(self, pretrained, requires_grad, input_size, hidden_size, num_layers, num_classes):
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

        ## this creates a lookup table where each camera gets its own learnable vector 
        ## format is dictionary: {camera_id: [list of 64 numbers]}
        # embedding_dim is the fingerprint/ signature for each camera, capturing specific characteristics, like viewing angle, etc 
        self.camera_embedding = nn.Embedding(num_cameras, embedding_dim)
        
        # NEW: Modified final layers to incorporate camera information
        # the fusion layer takes the image features + camera features and reduces dimensionality 
        # it's a linear/ affine transformation with this equation output = input × W^T + b
        self.fusion_layer = nn.Linear(2048 + embedding_dim, 512)

        ## add a drop layer / regularization to prevent overfitting
        self.dropout = nn.Dropout(0.2)

        ## finally predict the keypoints 
        self.output_layer = nn.Linear(512, 4)
        
        # Initialize camera embeddings
        ## assumes a normal distribution with mean 0 and std dev 0.1
        nn.init.normal_(self.camera_embedding.weight, 0, 0.1)
        #self.l0 = nn.Linear(2048, 4)  #### the second value is the number of points you want to predict

    def forward(self, x, camera_ids=None):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        #x = self.model.features(x)
        #x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)

        image_features = self.model.features(x)
        image_features = F.adaptive_avg_pool2d(image_features, 1).reshape(batch, -1)
        
        camera_embeddings = self.camera_embedding(camera_ids)
        combined_features = torch.cat([image_features, camera_embeddings], dim=1)
        x = F.relu(self.fusion_layer(combined_features))
        x = self.dropout(x)
        output = self.output_layer(x)
        # else:
        #     # ORIGINAL: Direct linear layer (fallback)
        #     output = self.l0(image_features)
        # l0 = self.l0(x)
        return output
