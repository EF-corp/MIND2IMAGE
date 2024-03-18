
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from typing import *

# class EEGDecoder(nn.Module):
#     def __init__(self, 
#                  in_feature: int = 840, 
#                  decoder_embedding_size: int = 1024, 
#                  additional_encoder_nhead: int = 8, 
#                  additional_encoder_dim_feedforward: int = 2048, 
#                  *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, batch_first=True)
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

#         self.fc = nn.Linear(in_feature, decoder_embedding_size)

#     def forward(self, 
#                 input_embeddings_batch: torch.Tensor, 
#                 #input_masks_invert: torch.Tensor
#                 ) -> torch.Tensor:


#         encoded_embedding = self.encoder(input_embeddings_batch)

#         encoded_embedding = F.relu(self.fc(encoded_embedding))
#         return encoded_embedding
    


class EEGfeatureExtracor(nn.Module):
    def __init__(self, 
                 text_encoder:nn.Module, 
                 image_encoder:nn.Module, 
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

    
    def forward(self, 
                texts: torch.Tensor, 
                images: torch.Tensor):
        text_feat  = self.text_encoder(texts)
        image_feat = self.image_encoder(images)

        text_embed  = F.normalize(text_feat, dim=-1)
        image_embed = F.normalize(image_feat, dim=-1)

        return text_embed, image_embed, text_feat, image_feat
    
class Image_encoder(nn.Module):
    def __init__(self, 
                 embedding_dim:int,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.image_embedding = resnet50(pretrained=True).to(device)
        self.weights = ResNet50_Weights.DEFAULT
        self.preprocess = self.weights.transforms().to(device)

        for param in self.image_embedding.parameters():
            param.requires_grad = False
        num_features = self.image_embedding.fc.in_features
        self.image_embedding.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_features, embedding_dim, bias=True),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=False),
        )

        self.image_embedding.fc.to(device)

    def forward(self,x: torch.Tensor):
        return self.image_embedding(x)
    
    def get_preprocess(self, image):
        return self.preprocess(image)

class EEGLSTM_Encoder(nn.Module):
    def __init__(self, 
                 in_channels:int=128, 
                 n_features:int=128, 
                 projection_dim:int=256, 
                 num_layers:int=4, 
                 device:Union[str,torch.device]=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(EEGLSTM_Encoder, self).__init__()
        self.hidden_size= n_features
        self.num_layers = num_layers
        self.device = device
        self.encoder = nn.LSTM(input_size=in_channels, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=n_features, out_features=projection_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        h_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        c_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        _, (h_n, c_n) = self.encoder( x, (h_n, c_n) )

        feat = h_n[-1]
        x = self.fc(feat)

        return x
    
class EEGTransformer(nn.Module):
    def __init__(self, 
                 input_dim:int=128,
                 n_features:int=512, 
                 n_heads:int=8, 
                 n_layers:int=6,
                 projection_dim:int=256, 
                 dropout:float=0.1):
        super(EEGTransformer, self).__init__()

        self.embedding = nn.Linear(in_features=input_dim, out_features=n_features)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=n_features, nhead=n_heads, dropout=dropout)

        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

        self.fc = nn.Linear(in_features=n_features, out_features=projection_dim, bias=False)

    def forward(self, x):

        x = self.embedding(x)

        x = x.permute(1, 0, 2)

        x = self.transformer(x)

        x = x.permute(1, 0, 2)
        x = self.fc(x[:, -1, :])
        return x
    

class CNN_block(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding, dropout) -> None:
        super().__init__()
        
        self.block = nn.Sequential(
                         nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                         nn.InstanceNorm2d(out_channels),
                         nn.LeakyReLU(),
                         nn.Dropout(p=dropout)
                         )
    def forward(self,x):
        return self.block(x)
    
class EEG_CNN(nn.Module):
    def __init__(self, 
                 input_shape:List[int]=[1, 440, 128],
                 projection_dim:int=256,
                 dropout:float=0.5, 
                 num_filters:List[int]=[128, 256, 512, 1024], 
                 kernel_sizes:List[int]=[3, 3, 3, 3], 
                 strides:List[int]=[2, 2, 2, 2], 
                 padding:List[int]=[1, 1, 1, 1]):
        super(EEG_CNN, self).__init__()

        self.layers = nn.ModuleList()
        in_channels = input_shape[0]
        for i, out_channels in enumerate(num_filters):
            self.layers.append(
                CNN_block(in_channels, out_channels, kernel_sizes[i], strides[i], padding[i], dropout)
                )

            in_channels = out_channels
        self.layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.fc = nn.Linear(num_filters[-1], projection_dim, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        x = F.normalize(x, dim=-1)
        return x
    
class EEG_resnet(nn.Module):
    def __init__(self,
                 init_channel:int=1,
                 resnet_pretrained:nn.Module=resnet50(pretrained=True),
                 fc_layer_param:int=256
                 ) -> None:
        super().__init__()
        self.conv_init = nn.Conv2d(in_channels=init_channel, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.resnet = resnet_pretrained
        self.resnet.fc = nn.Linear(in_features=2048, out_features=fc_layer_param, bias=False)

    def forward(self,x):

        x = self.conv_init(x)
        x = self.resnet(x)

        return x
    





# class EEG2ImageGan(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)





if __name__ == "__main__":
    # model = EEGfeatureExtracor(text_encoder=EEG_Encoder(), 
    #                            image_encoder=Image_encoder(embedding_dim=256))
    # eeg = torch.randn((512, 440, 128))
    # x = model.text_encoder(eeg)
    # print(x.shape)
    model = EEGTransformer()

    print(model)
    


    
