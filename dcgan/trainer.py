"""Boucle d'entraînement du DCGAN avec checkpoints et suivi TensorBoard."""

import copy
import time

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from .data import build_dataloader, denorm
from .models import build_models
from .utils import (get_device, latest_checkpoint, load_checkpoint, prune_checkpoints,
                    run_dirs, save_checkpoint, seed_everything)


class Trainer:
    def __init__(self, cfg: dict, resume=None):
        self.cfg = cfg
        seed_everything(cfg["seed"])
        self.device = get_device(cfg.get("device", "auto"))
        self.dirs = run_dirs(cfg)

        tcfg = cfg["train"]
        self.loader = build_dataloader(cfg["data"], pin_memory=self.device.type == "cuda")
        self.G, self.D = build_models(cfg["model"])
        self.G.to(self.device)
        self.D.to(self.device)

        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=tcfg["lr_g"], betas=(tcfg["beta1"], tcfg["beta2"]))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=tcfg["lr_d"], betas=(tcfg["beta1"], tcfg["beta2"]))
        self.criterion = nn.BCEWithLogitsLoss()

        # Bruit fixe pour visualiser l'évolution du générateur sur les mêmes z
        self.fixed_noise = torch.randn(tcfg["num_samples"], cfg["model"]["nz"], 1, 1, device=self.device)
        self.start_epoch = 0
        self.global_step = 0

        if resume:
            ckpt_path = latest_checkpoint(self.dirs["checkpoints"]) if resume == "latest" else resume
            if ckpt_path is None:
                print("Aucun checkpoint trouvé, démarrage à zéro.")
            else:
                self._resume(ckpt_path)

        self.writer = SummaryWriter(log_dir=str(self.dirs["tensorboard"]), purge_step=self.global_step or None)

    def _resume(self, path):
        ckpt = load_checkpoint(path, map_location=self.device)
        self.G.load_state_dict(ckpt["generator"])
        self.D.load_state_dict(ckpt["discriminator"])
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
        self.fixed_noise = ckpt["fixed_noise"].to(self.device)
        self.start_epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        print(f"Reprise depuis {path} (époque {self.start_epoch}, pas {self.global_step})")

    def _checkpoint(self, epoch):
        path = self.dirs["checkpoints"] / f"epoch_{epoch:04d}.pt"
        save_checkpoint(
            path,
            generator=self.G.state_dict(),
            discriminator=self.D.state_dict(),
            opt_g=self.opt_g.state_dict(),
            opt_d=self.opt_d.state_dict(),
            fixed_noise=self.fixed_noise.cpu(),
            epoch=epoch,
            global_step=self.global_step,
            config=copy.deepcopy(self.cfg),
        )
        prune_checkpoints(self.dirs["checkpoints"], self.cfg["output"].get("keep_last"))
        print(f"Checkpoint sauvegardé : {path}")

    @torch.no_grad()
    def _fixed_samples(self):
        self.G.eval()
        fake = self.G(self.fixed_noise)
        self.G.train()
        return vutils.make_grid(denorm(fake.cpu()), nrow=8, padding=2)

    def train_step(self, real):
        b = real.size(0)
        nz = self.cfg["model"]["nz"]
        real_target = 1.0 - self.cfg["train"].get("label_smoothing", 0.0)
        ones = torch.full((b,), real_target, device=self.device)
        zeros = torch.zeros(b, device=self.device)

        # --- Discriminateur : max log D(x) + log(1 - D(G(z)))
        noise = torch.randn(b, nz, 1, 1, device=self.device)
        fake = self.G(noise)
        out_real = self.D(real)
        out_fake = self.D(fake.detach())
        loss_d_real = self.criterion(out_real, ones)
        loss_d_fake = self.criterion(out_fake, zeros)
        loss_d = loss_d_real + loss_d_fake
        self.opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        self.opt_d.step()

        # --- Générateur : max log D(G(z))
        out_fake_g = self.D(fake)
        loss_g = self.criterion(out_fake_g, torch.ones(b, device=self.device))
        self.opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        self.opt_g.step()

        return {
            "loss/discriminator": loss_d.item(),
            "loss/generator": loss_g.item(),
            "score/D(x)": torch.sigmoid(out_real).mean().item(),
            "score/D(G(z))_before": torch.sigmoid(out_fake).mean().item(),
            "score/D(G(z))_after": torch.sigmoid(out_fake_g).mean().item(),
        }

    def fit(self):
        tcfg, ocfg = self.cfg["train"], self.cfg["output"]
        epochs = tcfg["epochs"]
        n_batches = len(self.loader)
        print(f"Device : {self.device} | images : {len(self.loader.dataset)} | "
              f"batches/époque : {n_batches} | époques : {self.start_epoch}->{epochs}")
        print(f"TensorBoard : tensorboard --logdir {self.dirs['tensorboard'].parent.parent}")

        if self.global_step == 0:
            real = next(iter(self.loader))[:64]
            self.writer.add_image("real_images", vutils.make_grid(denorm(real), nrow=8, padding=2), 0)
            # Époque 0 : générateur non entraîné, point de départ du GIF d'évolution
            grid = self._fixed_samples()
            vutils.save_image(grid, self.dirs["samples"] / "epoch_0000.png")
            self.writer.add_image("fake_images_epoch", grid, 0)

        last_saved = completed = self.start_epoch
        try:
            for epoch in range(self.start_epoch + 1, epochs + 1):
                t0 = time.time()
                for i, real in enumerate(self.loader, 1):
                    real = real.to(self.device, non_blocking=True)
                    metrics = self.train_step(real)
                    self.global_step += 1

                    if self.global_step % tcfg["log_every"] == 0:
                        for k, v in metrics.items():
                            self.writer.add_scalar(k, v, self.global_step)
                        print(f"[{epoch}/{epochs}][{i}/{n_batches}] "
                              f"Loss_D {metrics['loss/discriminator']:.4f}  Loss_G {metrics['loss/generator']:.4f}  "
                              f"D(x) {metrics['score/D(x)']:.3f}  "
                              f"D(G(z)) {metrics['score/D(G(z))_before']:.3f}/{metrics['score/D(G(z))_after']:.3f}")

                    if self.global_step % tcfg["sample_every"] == 0:
                        self.writer.add_image("fake_images", self._fixed_samples(), self.global_step)

                # Fin d'époque : grille sauvegardée sur disque (sert aussi au GIF d'évolution)
                grid = self._fixed_samples()
                vutils.save_image(grid, self.dirs["samples"] / f"epoch_{epoch:04d}.png")
                self.writer.add_image("fake_images_epoch", grid, epoch)
                self.writer.add_scalar("time/epoch_seconds", time.time() - t0, epoch)

                completed = epoch
                if epoch % ocfg["checkpoint_every"] == 0 or epoch == epochs:
                    self._checkpoint(epoch)
                    last_saved = epoch
        except KeyboardInterrupt:
            # Les poids actuels incluent l'époque en cours (partielle) ; on les sauvegarde
            # sous le numéro de la dernière époque complète pour pouvoir reprendre.
            print("\nInterruption.")
            if completed > last_saved:
                self._checkpoint(completed)
        finally:
            self.writer.close()
