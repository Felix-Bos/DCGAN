"""Création de GIFs de visages : interpolation dans l'espace latent et évolution de l'entraînement."""

from pathlib import Path

import torch
import torchvision.utils as vutils
from PIL import Image, ImageDraw

from .data import denorm
from .models import Generator
from .utils import get_device, load_checkpoint


def load_generator(checkpoint_path, device="auto"):
    """Recharge le générateur (et sa config) depuis un checkpoint d'entraînement."""
    device = get_device(device) if isinstance(device, str) else device
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    mcfg = ckpt["config"]["model"]
    G = Generator(mcfg["nz"], mcfg["ngf"], mcfg["nc"]).to(device)
    G.load_state_dict(ckpt["generator"])
    G.eval()
    return G, device


def slerp(z0, z1, t):
    """Interpolation sphérique : reste sur la « coquille » où vit le bruit gaussien."""
    a, b = z0.flatten(1), z1.flatten(1)
    omega = torch.acos((a / a.norm(dim=1, keepdim=True) * b / b.norm(dim=1, keepdim=True)).sum(1).clamp(-1, 1))
    so = torch.sin(omega)
    out = (torch.sin((1 - t) * omega) / so).unsqueeze(1) * a + (torch.sin(t * omega) / so).unsqueeze(1) * b
    # Repli sur l'interpolation linéaire si les vecteurs sont colinéaires
    t = t.unsqueeze(1)
    lin = (1 - t) * a + t * b
    out = torch.where((so.abs() < 1e-6).unsqueeze(1), lin, out)
    return out.view_as(z0)


def _to_pil(grid: torch.Tensor, scale: int = 1) -> Image.Image:
    arr = (grid.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(arr)
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    return img


def save_gif(frames, path, duration_ms=80, loop=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=loop, optimize=False)
    return path


@torch.no_grad()
def make_interpolation_gif(checkpoint_path, output_path, n_faces=16, n_keyframes=5,
                           steps_per_transition=20, nrow=None, duration_ms=60,
                           scale=2, seed=None, device="auto"):
    """GIF où chaque visage se transforme en un autre en parcourant l'espace latent.

    n_faces visages en grille ; chacun passe par n_keyframes vecteurs latents aléatoires,
    puis revient au premier pour une boucle sans saut.
    """
    G, device = load_generator(checkpoint_path, device)
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    keys = [torch.randn(n_faces, G.nz, 1, 1, generator=gen) for _ in range(n_keyframes)]
    keys.append(keys[0])  # boucle
    nrow = nrow or max(1, int(n_faces ** 0.5))

    frames = []
    for z0, z1 in zip(keys[:-1], keys[1:]):
        for s in range(steps_per_transition):
            t = torch.full((n_faces,), s / steps_per_transition)
            z = slerp(z0, z1, t).to(device)
            imgs = denorm(G(z)).cpu()
            frames.append(_to_pil(vutils.make_grid(imgs, nrow=nrow, padding=2), scale))
    return save_gif(frames, output_path, duration_ms)


@torch.no_grad()
def make_reveal_gif(checkpoint_path, output_path, n_faces=16, n_frames=40, hold_frames=20,
                    nrow=None, duration_ms=60, scale=2, seed=None, device="auto"):
    """GIF qui part d'un bruit blanc pur et fait émerger les visages générés.

    Chaque image mélange le visage et un bruit gaussien retiré à chaque frame :
    x = sqrt(a) * visage + sqrt(1 - a) * bruit, avec a qui passe de 0 (bruit seul) à 1 (visage net).
    """
    G, device = load_generator(checkpoint_path, device)
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    z = torch.randn(n_faces, G.nz, 1, 1, generator=gen).to(device)
    faces = G(z).cpu()  # dans [-1, 1]
    nrow = nrow or max(1, int(n_faces ** 0.5))

    frames = []
    for i in range(n_frames + 1):
        t = i / n_frames
        a = t * t * (3 - 2 * t)  # smoothstep : départ et arrivée en douceur
        noise = torch.randn(faces.shape, generator=gen)
        x = a ** 0.5 * faces + (1 - a) ** 0.5 * noise
        frames.append(_to_pil(vutils.make_grid(denorm(x), nrow=nrow, padding=2), scale))
    frames += [frames[-1]] * hold_frames  # on reste sur les visages finaux
    return save_gif(frames, output_path, duration_ms)


def make_training_gif(samples_dir, output_path, duration_ms=300, scale=1, label=True):
    """GIF de l'évolution des visages générés (bruit fixe) au fil des époques."""
    files = sorted(Path(samples_dir).glob("epoch_*.png"))
    if not files:
        raise FileNotFoundError(f"Aucune image epoch_*.png dans {samples_dir}")
    frames = []
    for f in files:
        img = Image.open(f).convert("RGB")
        if scale > 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        if label:
            draw = ImageDraw.Draw(img)
            text = f"epoch {int(f.stem.split('_')[1])}"
            draw.rectangle((0, 0, 8 + 7 * len(text), 16), fill=(0, 0, 0))
            draw.text((4, 2), text, fill=(255, 255, 255))
        frames.append(img)
    # On reste plus longtemps sur la dernière image
    frames += [frames[-1]] * 5
    return save_gif(frames, output_path, duration_ms)
