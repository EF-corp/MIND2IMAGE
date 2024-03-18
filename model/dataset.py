
import torch
import numpy as np
import os
from glob import glob
from typing import *
import cv2
from tqdm import tqdm

from natsort import natsorted

class EEG2ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset_path:str,
                 image_size:Tuple[int],
                 loaded_eeg_processing_model:torch.nn.Module,
                 resolution=None, # need for ada train
                 device:Union[str,torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 **kwargs) -> None:
        
        self.eegs   = []
        self.images = []
        self.labels = []
        self.class_name = []
        self.eeg_feat = []
        self.eeg_model = loaded_eeg_processing_model.to(device=device)

        for path in tqdm(sorted(glob(dataset_path))):
            loaded_array = np.load(path, allow_pickle=True)
            eeg = np.float32(loaded_array[1].T)
            self.eegs.append(eeg)
            img = np.float32(cv2.resize(loaded_array[0], image_size))
            self.images.append(np.transpose(img, (2, 0, 1)))
            self.labels.append(loaded_array[2])
            with torch.no_grad():
                norm = np.max(eeg) / 2.0
                eeg = (eeg - norm) / norm
                self.eeg_feat.append(self.eeg_model(torch.from_numpy(np.expand_dims(eeg, axis=0)).to(device)).detach().cpu().numpy()[0])
        self.eegs = torch.from_numpy(np.array(self.eegs)).to(torch.float32)
        self.images = torch.from_numpy(np.array(self.images)).to(torch.float32)
        self.eeg_feat = torch.from_numpy(np.array(self.eeg_feat)).to(torch.float32)
        self.labels = torch.from_numpy(np.array(self.labels)).to(torch.int32)

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):

        eeg = self.eegs[idx]
        norm = torch.max(eeg) / 2.0
        eeg = (eeg - norm) / norm
        image = self.images[idx]
        con = self.eeg_feat[idx]
        return image, con
    
    def get_label(self, idx):
        con = self.eeg_feat[idx]
        return con
    
class EEGFeatureDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path:str,
                 device:Union[str,torch.device]=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        x_eeg = []
        x_images = []
        labels = []

        self.device = device
        d = sorted(os.listdir(data_path))
        for path in tqdm(d[:len(d)//10]):
            loaded_array = np.load(os.path.join(data_path, path), allow_pickle=True)

            eeg_temp = loaded_array[1].T
            norm = np.max(eeg_temp)/2.0
            x_eeg.append((eeg_temp-norm)/norm)


            img = cv2.resize(loaded_array[0], (224, 224))
            img = np.float32(np.transpose(img, (2, 0, 1)))/255.0
            x_images.append(img)

            labels.append(loaded_array[2])

        self.eeg = torch.from_numpy(np.array(x_eeg)).float()
        self.labels = torch.from_numpy(np.array(labels)).long()
        self.images = torch.from_numpy(np.array(x_images)).float()

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return self.eeg[idx], self.images[idx], self.labels[idx]