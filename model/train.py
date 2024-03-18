
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from typing import *
from multiprocessing import cpu_count
import os
import json
import tempfile

from .utils import to_device, CustomError
from ..nvidia_ada import dnnlib
from ..nvidia_ada.training import training_loop
from ..nvidia_ada import legacy
# from ..nvidia_ada.torch_utils import *

class FeatureExtractorUser:
    def __init__(self,
                 train_data_path:str,
                 val_data_path:str,
                 dataset_class:torch.utils.data.Dataset,
                 out_dir:str,
                 batch_size:int,
                 epoch:int,
                 image_extractor_model:nn.Module,
                 eeg_extractor_model:nn.Module,
                 ImagEegModel:nn.Module,
                 #num_gpus:int,
                 scheduler:Any=None, # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_image, T_max=1024, eta_min=3e-4)
                 #gamma:float=0.999,
                 device:Union[str,torch.device]=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 nworkers:int=cpu_count(),
                 betas:Tuple[float]=(0.5, 0.999),
                 lr:float=3e-4,
                 init_epoch:int=0,
                 pretrain_path:Union[None,str]=None,
                 save_every:int=5,
                 model_name:str="eeg_model",
                 **kwargs) -> None:

        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.dataset_class = dataset_class


        self.batch_size = batch_size
        self.epoch = epoch
        self.image_extractor_model = image_extractor_model
        self.eeg_extractor_model = eeg_extractor_model
        self.device = device
        self.nworkers = nworkers
        self.betas = betas
        self.lr = lr
        self.init_epoch = init_epoch
        self.pretrain_path = pretrain_path
        self.save_every = save_every
        self.model_name = model_name



        self.train_dataloader = torch.utils.data.DataLoader(self.dataset_class(self.train_data_path, device=self.device),
                                                            num_workers=self.nworkers,
                                                            batch_size=self.batch_size)
        self.val_dataloader = torch.utils.data.DataLoader(self.dataset_class(self.val_data_path, device=self.device),
                                                          num_workers=self.nworkers,
                                                          batch_size=self.batch_size)

        self.model = ImagEegModel(self.eeg_extractor_model,
                                  self.image_extractor_model).to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.optim = optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas)

        if scheduler is not None:

            self.is_sched = True

            gamma = kwargs.get("gamma")
            stepsize = kwargs.get("stepsize")

            assert (gamma is not None or stepsize is not None), "You need to provide either a 'gamma' or a 'stepsize'"
            self.scheduler = scheduler(self.optim, gamma=gamma, step_size=stepsize)
        else:
            self.is_sched = False

        if self.pretrain_path is not None:
            print("Loading model from {}".format(self.pretrain_path))
            checkpoint = torch.load(self.pretrain_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])

            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.check_path = os.path.join(out_dir,"checkpoint")
        os.makedirs(self.check_path, exist_ok=True)


    def save_state_dict(self, e) -> None:

        model_state = {'model_state_dict': self.model.state_dict(),
                       'optim_state_dict': self.optim.state_dict()}

        if self.is_sched:
            model_state['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(
                    model_state,
                    os.path.join(self.check_path, f"{self.model_name}_{e}")
                )
    def cross_entropy(preds, targets, reduction='none'):

        assert reduction in ["none", "mean"], "! reduction must be 'none' or 'mean' !"

        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)

        return loss if reduction=="none" else loss.mean()


    def train(self, temperature:float=0.5) -> List[float]:

        self.model.train()

        loss_hist = []
        for epoch in range(self.init_epoch, self.epoch):
            running_loss = 0
            tq_dataloader = tqdm(self.train_dataloader)

            for batch_idx, data in enumerate(tq_dataloader):

                self.optim.zero_grad()

                EEGs, images, labels = to_device(data, device=self.device)

                images = self.image_extractor_model.get_preprocess(images)

                EEG_embed, image_embed, EEG_feat, image_feat = self.model(EEGs, images)

                logits = (EEG_embed @ image_embed.T) * torch.exp(torch.tensor(temperature))

                labels = torch.arange(image_embed.shape[0]).to(self.device)

                loss_i = self.criterion(logits, labels)
                loss_t = self.criterion(logits.T, labels)

                loss = (loss_i + loss_t) / 2.0
                loss = loss.mean()

                loss.backward()
                self.optim.step()

                running_loss += loss.item()

                tq_dataloader.set_description('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / (batch_idx+1.0)))
                if self.is_sched:
                    self.scheduler.step()
            loss_hist.append(running_loss)

            if epoch % self.save_every == 0:
                self.save_state_dict(e=epoch)

        return loss_hist

    @torch.no_grad()
    def validate(self, temperature:int=0.5):
        self.model.eval()
        running_loss = 0
        tq_dataloader = tqdm(self.val_dataloader)

        for batch_idx, data in enumerate(tq_dataloader):

            EEGs, images, labels = to_device(data, device=self.device)

            images = self.image_extractor_model.get_preprocess(images)

            EEG_embed, image_embed, EEG_feat, image_feat = self.model(EEGs, images)

            logits = (EEG_embed @ image_embed.T) * torch.exp(torch.tensor(temperature))
            images_similarity = image_embed @ image_embed.T
            EEGs_similarity = EEG_embed @ EEG_embed.T
            targets = F.softmax((images_similarity + EEGs_similarity) / 2 * temperature, dim=-1)

            images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
            EEGs_loss  = self.cross_entropy(logits, targets, reduction='none')

            loss = (images_loss + EEGs_loss) / 2.0
            running_loss += loss.mean().item()

            tq_dataloader.set_description('[%5d] loss: %.3f' % (batch_idx + 1, running_loss / (batch_idx+1.0)))


        print(f'Validation loss: {running_loss / len(self.val_dataloader)}')


        return running_loss


    #@torch.no_grad()
    def get_eeg_model(self, path_to_weiths_main_model) -> nn.Module:
        WEITHS = torch.load(path_to_weiths_main_model)
        self.model.load_state_dict(WEITHS['model_state_dict'])
        return self.model.text_encoder

    @torch.no_grad()
    def generate(self, image, eeg):
        image, eeg = to_device([image, eeg], device=self.device)

        image = self.image_extractor_model.get_preprocess(image)

        EEG_embed, image_embed, EEG_feat, image_feat = self.model(eeg, image)
        return EEG_embed, image_embed

    @torch.no_grad()
    def generate_eeg(self,
                     eeg:torch.Tensor,
                     path_to_weiths_main_model:str):

        model = self.get_eeg_model(path_to_weiths_main_model=path_to_weiths_main_model)
        eeg = model(eeg)
        return F.normalize(eeg, dim=-1)


class ADAUser:
    """custom Trainer for nvidia ADA model"""
    def __init__(self,
                 dataset_name: str = "dataset.EEG2ImageDataset",
                 gpus: int = 1, 
                 snap: int = 10, 
                 metrics: Any = None, 
                 seed: int = 0, 
                 data: Any = None, 
                 cond: bool = False, 
                 subset: Any = None, 
                 mirror: bool = False, 
                 cfg: str = 'auto', 
                 gamma: Any = None, 
                 kimg: Any = None, 
                 batch: Any = None, 
                 aug: str = 'ada', 
                 p: Any = None, 
                 target: Any = None, 
                 augpipe: str = 'bgc', 
                 resume: str = 'noresume', 
                 freezed: int = 0, 
                 fp32: bool = False, 
                 nhwc: bool = False, 
                 allow_tf32: bool = False, 
                 nobench: bool = False,
                 workers: int = cpu_count) -> None:
        
        self.args = dnnlib.EasyDict()
        self.args.num_gpus = gpus
        self.args.image_snapshot_ticks = snap
        self.args.network_snapshot_ticks = snap

        self.args.metrics = metrics
        self.args.random_seed = seed
        self.args.training_set_kwargs = dnnlib.EasyDict(class_name=dataset_name, 
                                                   path=data, 
                                                   use_labels=True, 
                                                   max_size=None, 
                                                   xflip=False)
        self.args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=8, prefetch_factor=2)
        try:
            training_set = dnnlib.util.construct_class_by_name(**self.args.training_set_kwargs)
            self.args.training_set_kwargs.resolution = training_set.resolution
            self.args.training_set_kwargs.use_labels = training_set.has_labels
            self.args.training_set_kwargs.max_size = len(training_set)
            del training_set 

        except IOError as err:
            raise CustomError(f'--data: {err}')
        
        if cond and not self.args.training_set_kwargs.use_labels:

            raise CustomError('--cond=True requires labels specified in dataset.json')
        
        else:
            self.args.training_set_kwargs.use_labels = False

        if subset is not None:
            assert isinstance(subset, int)
            if not 1 <= subset <= self.args.training_set_kwargs.max_size:
                raise CustomError(f'--subset must be between 1 and {self.args.training_set_kwargs.max_size}')
            if subset < self.args.training_set_kwargs.max_size:
                self.args.training_set_kwargs.max_size = subset
                self.args.training_set_kwargs.random_seed = self.args.random_seed

        if mirror:
            self.args.training_set_kwargs.xflip = True

        cfg_specs = {
            'auto': dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
            'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
            'paper256': dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
            'paper512': dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
            'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
            'cifar': dict(ref_gpus=2,  kimg=5500, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
        }
        spec = dnnlib.EasyDict(cfg_specs[cfg])
        if cfg == 'auto':
            desc += f'{gpus:d}'
            spec.ref_gpus = gpus
            res = self.args.training_set_kwargs.resolution
            spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus)
            spec.mbstd = min(spec.mb // gpus, 4)
            spec.fmaps = 1 if res >= 512 else 0.5
            spec.lrate = 0.002 if res >= 1024 else 0.0025
            spec.gamma = 0.0002 * (res ** 2) / spec.mb 

            spec.ema = spec.mb * 10 / 32

        self.args.G_kwargs = dnnlib.EasyDict(class_name='..nvidia_ada.training.networks.Generator', 
                                             z_dim=512, w_dim=512, 
                                             mapping_kwargs=dnnlib.EasyDict(), 
                                             synthesis_kwargs=dnnlib.EasyDict())
        self.args.D_kwargs = dnnlib.EasyDict(class_name='..nvidia_ada.training.networks.Discriminator', 
                                             block_kwargs=dnnlib.EasyDict(), 
                                             mapping_kwargs=dnnlib.EasyDict(), 
                                             epilogue_kwargs=dnnlib.EasyDict())
        self.args.G_kwargs.synthesis_kwargs.channel_base = self.args.D_kwargs.channel_base = int(spec.fmaps * 32768)
        self.args.G_kwargs.synthesis_kwargs.channel_max = self.args.D_kwargs.channel_max = 512
        self.args.G_kwargs.mapping_kwargs.num_layers = spec.map
        self.args.G_kwargs.synthesis_kwargs.num_fp16_res = self.args.D_kwargs.num_fp16_res = 4 
        self.args.G_kwargs.synthesis_kwargs.conv_clamp = self.args.D_kwargs.conv_clamp = 256 
        self.args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd

        self.args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
        self.args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
        self.args.loss_kwargs = dnnlib.EasyDict(class_name='..nvidia_ada.training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)
        self.args.total_kimg = spec.kimg
        self.args.batch_size = spec.mb
        self.args.batch_gpu = spec.mb // spec.ref_gpus
        self.args.ema_kimg = spec.ema
        self.args.ema_rampup = spec.ramp

        if cfg == 'cifar':
            self.args.loss_kwargs.pl_weight = 0
            self.args.loss_kwargs.style_mixing_prob = 0
            self.args.D_kwargs.architecture = 'orig'

        if gamma is not None:
            self.args.loss_kwargs.r1_gamma = gamma

        if kimg is not None:
            self.args.total_kimg = kimg

        if batch is not None:
            self.args.batch_size = batch
            self.args.batch_gpu = batch // gpus

        if aug == 'ada':
            self.args.ada_target = 0.6

        if p is not None:
            self.args.augment_p = p
        if target is not None:
            self.args.ada_target = target


        augpipe_specs = {
            'blit':   dict(xflip=1, rotate90=1, xint=1),
            'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
            'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
            'filter': dict(imgfilter=1),
            'noise':  dict(noise=1),
            'cutout': dict(cutout=1),
            'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
            'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
            'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
            'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
            'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
        }

        if aug != 'noaug':
            self.args.augment_kwargs = dnnlib.EasyDict(class_name='..nvidia_ada.training.augment.AugmentPipe', **augpipe_specs[augpipe])

        resume_specs = {
            'ffhq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
            'ffhq512': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
            'ffhq1024': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
            'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
            'lsundog256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
        }

        if resume is None:
            resume = 'noresume'
        elif resume == 'noresume':
            desc += '-noresume'
        elif resume in resume_specs:
            desc += f'-resume{resume}'
            self.args.resume_pkl = resume_specs[resume]
        else:
            self.args.resume_pkl = resume

        if resume != 'noresume':
            self.args.ada_kimg = 100
            self.args.ema_rampup = None 

        if freezed is not None:
            self.args.D_kwargs.block_kwargs.freeze_layers = freezed

        if fp32:
            self.args.G_kwargs.synthesis_kwargs.num_fp16_res = self.args.D_kwargs.num_fp16_res = 0
            self.args.G_kwargs.synthesis_kwargs.conv_clamp = self.args.D_kwargs.conv_clamp = None

        if nhwc:
            self.args.G_kwargs.synthesis_kwargs.fp16_channels_last = self.args.D_kwargs.block_kwargs.fp16_channels_last = True

        self.args.cudnn_benchmark = not nobench
        self.args.allow_tf32 = allow_tf32
        self.args.data_loader_kwargs.num_workers = workers       
        

    def subprocess_fn(rank, args, temp_dir):
        if args.num_gpus > 1:
            init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)

        training_loop.training_loop(rank=rank, **args)
        
    def run_train(self, outdir, namesec):
        self.args.run_dir = os.path.join(outdir, namesec)

        os.makedirs(self.args.run_dir, exist_ok=True)
        with open(os.path.join(self.args.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(self.args, f, indent=2)
        try:
            torch.multiprocessing.set_start_method('spawn')
        except:
            pass
        with tempfile.TemporaryDirectory() as temp_dir:
            if self.args.num_gpus == 1:
                self.subprocess_fn(rank=0, args=self.args, temp_dir=temp_dir)
            else:
                torch.multiprocessing.spawn(fn=self.subprocess_fn, args=(self.args, temp_dir), nprocs=self.args.num_gpus)

    # @torch.no_grad()
    # def generate_image_from_eeg(self,
    #                             model_pkl_path:str,
    #                             data:str,
    #                             batch:int,
    #                             out_dir:str="./generate",
    #                             seeds:Union[Iterator[int], List, Tuple]=(0,),
    #                             device:Union[str,torch.device]=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> str:
        
    #     with dnnlib.util.open_url(model_pkl_path) as f:
    #         G = legacy.load_network_pkl(f)['G_ema'].to(device)

    #     os.makedirs(out_dir, exist_ok=True)








if __name__ == "__main__":
    pass