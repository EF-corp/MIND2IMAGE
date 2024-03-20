import yaml
import argparse
import torch
from multiprocessing import cpu_count

def get_arguments():
    parser = argparse.ArgumentParser(description='config for training model')

    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--val_data_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epoch', type=int, required=True)

    parser.add_argument('--image_height', type=int, default=128)
    parser.add_argument('--image_width', type=int, default=128)
    parser.add_argument('--input_channel', type=int, default=3)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--padding', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--n_classes', type=int, default=40)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--out_size', type=int, default=256)
    parser.add_argument('--nworkers', type=int, default=cpu_count()-1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--init_epoch', type=int, default=0)
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--model_name', type=str, default='eeg_model')

    args = parser.parse_args()
    return args


def args2yaml(args, yaml_file):

    with open(yaml_file, 'w') as file:
        yaml.dump(vars(args), file, default_flow_style=False)


if __name__ == "__main__":
    args2yaml(get_arguments(), "config.yml")