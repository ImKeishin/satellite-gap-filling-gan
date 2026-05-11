from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_l1: float = 100.0,
) -> Tuple[float, float]:
    adversarial_loss = nn.MSELoss()
    pixelwise_loss = nn.L1Loss()

    generator.train()
    discriminator.train()

    total_g_loss = 0.0
    total_d_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        masked_input = batch["masked_input"].to(device)
        target = batch["target"].to(device)

        g_optimizer.zero_grad(set_to_none=True)
        generated = generator(masked_input)
        pred_fake = discriminator(masked_input, generated)
        valid = torch.ones_like(pred_fake, device=device)
        g_adv = adversarial_loss(pred_fake, valid)
        g_l1 = pixelwise_loss(generated, target)
        g_loss = g_adv + lambda_l1 * g_l1
        g_loss.backward()
        g_optimizer.step()

        d_optimizer.zero_grad(set_to_none=True)
        pred_real = discriminator(masked_input, target)
        valid = torch.ones_like(pred_real, device=device)
        loss_real = adversarial_loss(pred_real, valid)
        pred_fake_detached = discriminator(masked_input, generated.detach())
        fake = torch.zeros_like(pred_fake_detached, device=device)
        loss_fake = adversarial_loss(pred_fake_detached, fake)
        d_loss = 0.5 * (loss_real + loss_fake)
        d_loss.backward()
        d_optimizer.step()

        total_g_loss += float(g_loss.item())
        total_d_loss += float(d_loss.item())

    return total_g_loss / len(dataloader), total_d_loss / len(dataloader)


@torch.no_grad()
def validate_one_epoch(generator: nn.Module, dataloader, device: torch.device) -> float:
    generator.eval()
    criterion = nn.L1Loss()
    total = 0.0
    for batch in tqdm(dataloader, desc="Validation", leave=False):
        masked_input = batch["masked_input"].to(device)
        target = batch["target"].to(device)
        generated = generator(masked_input)
        total += float(criterion(generated, target).item())
    return total / len(dataloader)


def save_checkpoint(
    path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "g_optimizer": g_optimizer.state_dict(),
            "d_optimizer": d_optimizer.state_dict(),
            "history": history,
        },
        path,
    )
