import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from darkdiff.data import PairDataset, get_pairs
from darkdiff.model import SD_MODEL, DarkDiff, empty_prompt_embeds, load_pretrained
from darkdiff.vae import ResidualVAE


def main(argv=None):
    p = argparse.ArgumentParser(description="Stage 2: train the DarkDiff latent diffusion model.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset", default="sid", choices=["sid", "lrd"])
    p.add_argument("--model-id", default=SD_MODEL)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--crop", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr-new", type=float, default=2.5e-4)
    p.add_argument("--lr-unet", type=float, default=5e-5)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--region-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="runs/ldm")
    p.add_argument("--save-every", type=int, default=100)
    args = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    unet, sched = load_pretrained(args.model_id)
    rvae = ResidualVAE.from_pretrained(args.model_id)
    rvae.load_state_dict(torch.load(args.vae_ckpt, map_location="cpu", weights_only=True))
    model = DarkDiff(unet, rvae, sched, empty_prompt_embeds(args.model_id), region_size=args.region_size)
    model = model.to(dev).train()
    opt = torch.optim.Adam(model.param_groups(args.lr_new, args.lr_unet))

    dl = DataLoader(PairDataset(get_pairs(args.dataset, args.data_root, "train"), args.crop, train=True),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    amp = torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")
    for epoch in tqdm(range(1, args.epochs + 1), desc="epochs"):
        pbar = tqdm(dl, desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            with amp:
                loss, logs = model.training_loss(batch["noisy"].to(dev), batch["clean"].to(dev),
                                                 batch["srgb"].to(dev), lam=args.lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix(logs)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(model.trainable_state_dict(), out / f"ldm_epoch{epoch}.pt")
            torch.save(model.trainable_state_dict(), out / "ldm_last.pt")


if __name__ == "__main__":
    main()
