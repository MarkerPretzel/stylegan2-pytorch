import argparse

import torch
from torchvision import utils

from model import Generator

import os
import imageio
import numpy as np
from PIL import Image


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-i", "--index", type=int, default=0, help="index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2 checkpoints")
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=7, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="factor",
        help="filename prefix to result samples",
    )
    parser.add_argument(
        "factor",
        type=str,
        help="name of the closed form factorization result factor file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="fps of the video",
    )
    parser.add_argument(
        "--sec",
        type=int,
        default=1,
        help="duration of the video in seconds (integer)",
    )
    parser.add_argument(
        "--dir_path",
        type=str,
        default='/kaggle/working/stylegan2-pytorch',
        help="path to images",
    )

    args = parser.parse_args()

    eigvec = torch.load(args.factor)["eigvec"].to(args.device)
    ckpt = torch.load(args.ckpt)
    g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier).to(args.device)
    g.load_state_dict(ckpt["g_ema"], strict=False)

    trunc = g.mean_latent(4096)

    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(latent)
   
    num_frames = args.fps * args.sec
    degree_per_frame = (args.degree * 2) / num_frames
    degree = - args.degree
    images = []
    for frame in range(num_frames):
        direction = degree * eigvec[:, args.index].unsqueeze(0)
        img, _ = g(
        [latent + direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
        )    
        print(img.shape)
        ndarr = img.mul(255).add_(0.5).clamp_(0, 255).squeeze(dim=0).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        #im.save(f"index-{args.index}_degree-{degree}_{args.out_prefix}.png")
        images.append(im)
        
    ### The following code is copied from https://dev.to/slushnys/how-to-create-a-video-from-an-image-with-python-26p5 ###
    #for f in os.listdir(args.dir_path):
        #if f.endswith('png'):
            #images.append(f)
        
    write_to = 'output/{}.mp4'.format(args.out_prefix) # have a folder of output where output files could be stored.

    writer = imageio.get_writer(write_to, format='mp4', mode='I', fps=args.fps)

    for i in range(num_frames):
        print(img.size)
        writer.append_data(np.asarray(images[i]))
    writer.close()
