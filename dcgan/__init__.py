"""DCGAN pour la génération de visages (CelebA)."""

from .data import CelebADataset, build_dataloader, denorm
from .gif import make_interpolation_gif, make_reveal_gif, make_training_gif
from .models import Discriminator, Generator
