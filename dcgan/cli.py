"""Interface en ligne de commande : python -m dcgan {train,sample,gif} ..."""

import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils


def cmd_train(args):
    from .trainer import Trainer
    from .utils import load_config, set_override

    cfg = load_config(args.config)
    for ov in args.set or []:
        key, _, value = ov.partition("=")
        set_override(cfg, key, value)
    Trainer(cfg, resume=args.resume).fit()


@torch.no_grad()
def cmd_sample(args):
    from .data import denorm
    from .gif import load_generator

    G, device = load_generator(args.checkpoint, args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    z = torch.randn(args.n, G.nz, 1, 1, device=device)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(denorm(G(z)).cpu(), out, nrow=args.nrow or int(args.n ** 0.5), padding=2)
    print(f"Images sauvegardées : {out}")


def cmd_gif(args):
    from .gif import make_interpolation_gif, make_reveal_gif, make_training_gif

    if args.mode in ("interp", "reveal") and not args.checkpoint:
        raise SystemExit(f"--checkpoint est requis pour le mode {args.mode}")
    if args.mode == "reveal":
        path = make_reveal_gif(
            args.checkpoint, args.output, n_faces=args.n, n_frames=args.frames,
            hold_frames=args.hold, duration_ms=args.duration, scale=args.scale,
            seed=args.seed, device=args.device,
        )
    elif args.mode == "interp":
        path = make_interpolation_gif(
            args.checkpoint, args.output, n_faces=args.n, n_keyframes=args.keyframes,
            steps_per_transition=args.steps, duration_ms=args.duration, scale=args.scale,
            seed=args.seed, device=args.device,
        )
    else:
        if not args.samples_dir:
            raise SystemExit("--samples-dir est requis pour le mode training")
        path = make_training_gif(args.samples_dir, args.output, duration_ms=args.duration, scale=args.scale)
    print(f"GIF sauvegardé : {path}")


def build_parser():
    p = argparse.ArgumentParser(prog="dcgan", description="DCGAN sur les visages CelebA")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Lancer / reprendre un entraînement")
    t.add_argument("-c", "--config", default="configs/celeba.yaml")
    t.add_argument("--resume", nargs="?", const="latest", default=None,
                   help="Reprendre depuis un checkpoint (chemin, ou sans valeur = le plus récent)")
    t.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="Surcharger la config, ex. --set train.epochs=5 --set data.max_images=10000")
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("sample", help="Générer une grille de visages depuis un checkpoint")
    s.add_argument("checkpoint")
    s.add_argument("-o", "--output", default="outputs/samples.png")
    s.add_argument("-n", type=int, default=64)
    s.add_argument("--nrow", type=int, default=None)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_sample)

    g = sub.add_parser("gif", help="Créer un GIF de visages")
    g.add_argument("mode", choices=["reveal", "interp", "training"],
                   help="reveal : du bruit blanc aux visages ; interp : morphing dans l'espace latent ; "
                        "training : évolution au fil des époques")
    g.add_argument("--checkpoint", help="(reveal/interp) checkpoint .pt du générateur")
    g.add_argument("--samples-dir", help="(training) dossier runs/<exp>/samples")
    g.add_argument("-o", "--output", default="outputs/faces.gif")
    g.add_argument("-n", type=int, default=16, help="(reveal/interp) nombre de visages dans la grille")
    g.add_argument("--frames", type=int, default=40, help="(reveal) nombre d'images du bruit au visage")
    g.add_argument("--hold", type=int, default=20, help="(reveal) images figées sur le résultat final")
    g.add_argument("--keyframes", type=int, default=5, help="(interp) nombre de points latents visités")
    g.add_argument("--steps", type=int, default=20, help="(interp) images entre deux points latents")
    g.add_argument("--duration", type=int, default=60, help="durée d'une image en ms")
    g.add_argument("--scale", type=int, default=2, help="facteur d'agrandissement")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--device", default="auto")
    g.set_defaults(func=cmd_gif)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
