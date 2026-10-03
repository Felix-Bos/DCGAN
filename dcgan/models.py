"""Architectures du générateur et du discriminateur (Radford et al., 2015)."""

import torch.nn as nn


def weights_init(m):
    """Initialisation DCGAN : N(0, 0.02) pour les convolutions, N(1, 0.02) pour BatchNorm."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Generator(nn.Module):
    """z (nz x 1 x 1) -> image (nc x 64 x 64) dans [-1, 1]."""

    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.nz = nz
        self.main = nn.Sequential(
            # nz x 1 x 1 -> (ngf*8) x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # -> (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # -> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # -> ngf x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # -> nc x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """image (nc x 64 x 64) -> logit (pas de sigmoïde : on utilise BCEWithLogitsLoss)."""

    def __init__(self, ndf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            # nc x 64 x 64 -> ndf x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # -> 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x).view(-1)


def build_models(model_cfg: dict):
    G = Generator(model_cfg["nz"], model_cfg["ngf"], model_cfg["nc"])
    D = Discriminator(model_cfg["ndf"], model_cfg["nc"])
    G.apply(weights_init)
    D.apply(weights_init)
    return G, D
