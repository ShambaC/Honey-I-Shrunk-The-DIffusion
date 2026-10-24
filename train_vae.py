import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from darkdiff.data import PairDataset, get_pairs
from darkdiff.model import SD_MODEL
from darkdiff.vae import ResidualVAE, VAELoss


def main(argv=None):
    p = argparse.ArgumentParser(description="Stage 1: fine-tune the residual VAE (noisy LRGB -> clean sRGB).")
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset", default="sid", choices=["sid", "lrd"])
    p.add_argument("--model-id", default=SD_MODEL)
    p.add_argument("--crop", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="runs/vae")
    p.add_argument("--save-every", type=int, default=100)
    args = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rvae = ResidualVAE.from_pretrained(args.model_id).to(dev).train()
    loss_fn = VAELoss().to(dev).train()
    betas = (0.5, 0.9)
    opt_g = torch.optim.Adam(list(rvae.parameters()) + [loss_fn.logvar], lr=args.lr, betas=betas)
    opt_d = torch.optim.Adam(loss_fn.disc.parameters(), lr=args.lr, betas=betas)

    dl = DataLoader(PairDataset(get_pairs(args.dataset, args.data_root, "train"), args.crop, train=True),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    amp = torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")
    step = 0
    for epoch in tqdm(range(1, args.epochs + 1), desc="epochs"):
        pbar = tqdm(dl, desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            x, target = batch["noisy"].to(dev), batch["srgb"].to(dev)
            with amp:
                rec, post = rvae(x)
                g_loss, glogs = loss_fn.generator_loss(target, rec, post, step, rvae.last_layer())
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            # the generator backward also filled discriminator grads, so clear them before the disc step
            with amp:
                d_loss, dlogs = loss_fn.discriminator_loss(target, rec, step)
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            step += 1
            pbar.set_postfix({**glogs, **dlogs})
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(rvae.state_dict(), out / f"vae_epoch{epoch}.pt")
            torch.save(rvae.state_dict(), out / "vae_last.pt")


if __name__ == "__main__":
    main()
