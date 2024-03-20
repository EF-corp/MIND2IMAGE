import torch
import torch.nn as nn
import numpy as np
from typing import *
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pickle
from scipy.optimize import linear_sum_assignment

from sklearn.cluster import KMeans


def to_device(*data:List[Union[torch.Tensor, torch.nn.Module]], 
              device:Union[str,torch.device]=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    return list(map(lambda el: el.to(device), data))

def setup_snapshot_image_grid(test_set, random_seed):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // test_set.image_shape[2], 8, 32)
    gh = np.clip(5120 // test_set.image_shape[1], 5, 40)

    if not test_set.has_labels:
        all_indices = list(range(len(test_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        label_groups = dict()
        for idx in range(len(test_set)):
            label = tuple([test_set.labels[idx].detach().cpu().item()])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    images, labels = zip(*[test_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

class CustomDataset(Dataset):
    def __init__(self, eegs, images, labels):
        self.eegs = eegs
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        eeg = self.eegs[index]
        image = self.images[index]
        label = self.labels[index]
        return eeg, image, label

    def __len__(self):
        return len(self.eegs)

class EEGDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

    def __iter__(self):
        batch_eeg = []
        batch_image = []
        labels_in_batch = []
        for eeg, image, label in super().__iter__():
            eeg = torch.squeeze(eeg, dim=0)
            image = torch.squeeze(image, dim=0)
            label = torch.squeeze(label, dim=0)
            if not batch_eeg:
                batch_eeg.append(eeg)
                batch_image.append(image)
                labels_in_batch.append(label)
                continue

            if label in labels_in_batch:
                continue

            batch_eeg.append(eeg)
            batch_image.append(image)
            labels_in_batch.append(label)

            if len(batch_eeg) == self.batch_size:
                yield torch.stack(batch_eeg), torch.stack(batch_image), torch.tensor(labels_in_batch)
                batch_eeg   = []
                batch_image = []
                labels_in_batch = []
        
        if batch_eeg:
            yield torch.stack(batch_eeg), torch.stack(batch_image), torch.tensor(labels_in_batch)

# class EEGOnly(Dataset):
#     def __init__(self, 
#                  data:str, 
#                  eeg_model:nn.Module, 
#                  device:Union[str,torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
#         self.eeg_model = eeg_model
        

#     def __getitem__(self, index):
#         eeg = self.eegs[index]
#         image = self.images[index]
#         label = self.labels[index]
#         return eeg, image, label

#     def __len__(self):
#         return len(self.eegs)

def cross_entropy(preds, 
                  targets, 
                  reduction:str='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)

    return loss if reduction == 'none' else loss.mean()

class CustomError(Exception):
    pass

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        Image.fromarray(img, 'RGB').save(fname)


# class ModelLoader:
#     def __init__(self) -> None:
#         pass

#     def load_pt_model(model:nn.Module, path2data:str,namekey:Any=None) -> nn.Module:
#         WEITHS = torch.load(path2data)
#         if namekey is not None:
#             model.load_state_dict(WEITHS[namekey])
#         else:
#             try:
#                 model.load_state_dict(WEITHS)
#             except Exception as e:
#                 print(e)
#                 return model
#         return model
#     def load_pkl_model(model:nn.Module, path2data:str):
#         data = pickle.Unpickler(path2data).load()
            
class K_means:
    def __init__(self, n_clusters=39, random_state=45):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
    def transform(self, text_embed, image_embed, Y_text=None, Y_image=None):
        
        text_label = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, init='k-means++').fit_predict(text_embed)
        text_score = self.cluster_acc(Y_text, text_label)

        image_label = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, init='k-means++').fit_predict(image_embed)
        image_score = self.cluster_acc(Y_image, image_label)

        return (text_label, image_label), (text_score, image_score)
    
    
    def cluster_acc(self, y_true, y_pred):

        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size
            
def get_eeg_model(model, path_to_weiths_main_model) -> nn.Module:
    WEITHS = torch.load(path_to_weiths_main_model)
    model.load_state_dict(WEITHS['model_state_dict'])
    return model.text_encoder