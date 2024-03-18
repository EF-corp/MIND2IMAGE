import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import *
import hashlib
import warnings
import urllib
from tqdm import tqdm
import numpy as np
from ..mind_reader.nvidia_ada import legacy
from ..mind_reader.nvidia_ada import dnnlib


def download(url: str, 
             root: str) -> str:
    
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def load_model(model:nn.Module,
               path_or_url:str,
               to_this_model:bool=False,
               root:str|None=None,
               device:Union[str,torch.device]=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    if os.path.exists(path=path_or_url):

        WEITHS = torch.load(path_or_url)
        if not to_this_model:
            model = model.load_state_dict(WEITHS['model_state_dict'])
            return model.to(device)
        else:
            model.state_dict(WEITHS).to(device).copy_to(model.state_dict())

    else:
        path2loaded = download(url=path_or_url, root=root)
        WEITHS = torch.load(path2loaded)

        if not to_this_model:
            model = model.load_state_dict(WEITHS['model_state_dict'])
            return model.to(device)
        
        else:
            model.state_dict(WEITHS).to(device).copy_to(model.state_dict())



@torch.no_grad()
def generate(eeg_preprocess_model:nn.Module,
             path_ada:str,
             outdir:str,
             data:Union[Tuple, List],
             
             device:Union[str,torch.device]=torch.device("cuda" if torch.cuda.is_available() else "cpu"),):
    
    os.makedirs(outdir, exist_ok=True)

    eeg, image, label = data
    norm = np.max(eeg) / 2.0
    eeg = (eeg - norm) / norm
    processed_eeg = eeg_preprocess_model(torch.from_numpy(np.expand_dims(eeg, axis=0)).to(device)).detach().cpu().numpy()[0]
    processed_eeg = torch.from_numpy(processed_eeg).to(torch.float32)

    with dnnlib.util.open_url(path_ada) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    grid_z = torch.randn((processed_eeg.shape[0], G.z_dim), device=device)
    grid_c = torch.from_numpy(processed_eeg).to(device)

    gen_images = torch.cat([G(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()

    return gen_images



if __name__ == "__main__":
    pass

