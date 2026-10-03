"""Chargement du dataset CelebA (images alignées) pour le DCGAN."""

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = {"train": 0, "val": 1, "test": 2}


def build_transform(image_size: int) -> transforms.Compose:
    """Redimensionne, recadre au centre et normalise les images dans [-1, 1]."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def denorm(x):
    """Passe un tenseur de [-1, 1] à [0, 1]."""
    return ((x + 1) / 2).clamp(0, 1)


class CelebADataset(Dataset):
    """Dataset plat : un dossier d'images, avec filtrage optionnel par partition."""

    def __init__(self, root, image_size=64, partition_file=None, split="all", max_images=None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dossier d'images introuvable : {self.root}")

        if partition_file and split != "all":
            if split not in SPLITS:
                raise ValueError(f"split invalide '{split}', attendu : {list(SPLITS)} ou 'all'")
            wanted = SPLITS[split]
            files = []
            with open(partition_file) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2 and int(parts[1]) == wanted:
                        files.append(parts[0])
        else:
            files = sorted(p.name for p in self.root.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)

        if max_images:
            files = files[:max_images]
        if not files:
            raise RuntimeError(f"Aucune image trouvée dans {self.root} (split={split})")

        self.files = files
        self.transform = build_transform(image_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.root / self.files[idx]).convert("RGB")
        return self.transform(img)


def build_dataloader(data_cfg: dict, pin_memory: bool = False) -> DataLoader:
    dataset = CelebADataset(
        root=data_cfg["root"],
        image_size=data_cfg["image_size"],
        partition_file=data_cfg.get("partition_file"),
        split=data_cfg.get("split", "all"),
        max_images=data_cfg.get("max_images"),
    )
    num_workers = data_cfg.get("num_workers", 0)
    return DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
