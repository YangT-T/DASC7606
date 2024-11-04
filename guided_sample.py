import torch
import torch.nn as nn
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import Diffuser
import math
import argparse
import datetime
from pathlib import Path
from tqdm import tqdm


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--n_samples",
        type=int,
        help="define sampling amounts after every epoch trained",
        default=36,
    )
    parser.add_argument(
        "--model_base_dim", type=int, help="base dim of Unet", default=64
    )
    parser.add_argument(
        "--timesteps", type=int, help="sampling steps of DDPM", default=1000
    )
    parser.add_argument(
        "--model_ema_steps", type=int, help="ema model evaluation interval", default=10
    )
    parser.add_argument(
        "--model_ema_decay", type=float, help="ema model decay", default=0.995
    )
    parser.add_argument("--cpu", action="store_true", help="cpu training")

    args = parser.parse_args()

    return args


def main(args):
    device = "cpu" if args.cpu else "cuda"
    model = Diffuser(
        timesteps=args.timesteps,
        image_size=28,
        in_channels=1,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4],
    ).to(device)

    # create project name with current time
    exp_name = datetime.datetime.now().strftime("%m%d-%H%M%S")
    exp_path = Path("logs") / exp_name
    exp_path.mkdir(parents=True)
    (exp_path / "ckpt").mkdir(exist_ok=True)
    (exp_path / "img").mkdir(exist_ok=True)

    # point=torch.load("./logs/1021-130907/ckpt/99.pt")
    point = torch.load("asdfsa")
    # model.load_state_dict(point["model"])
    model.load_state_dict(point["model_ema"])

    for y in range(10):
        model.eval()
        samples = model.module.guided_sampling(args.n_samples, y, device=device)
        save_image(
            samples, exp_path / "img" / f"{y}.png", nrow=int(math.sqrt(args.n_samples))
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
