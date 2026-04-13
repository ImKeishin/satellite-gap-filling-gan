import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(
    generator,
    discriminator,
    dataloader,
    g_optimizer,
    d_optimizer,
    device,
    lambda_l1=100.0,
):
    adversarial_loss = nn.MSELoss()
    pixelwise_loss = nn.L1Loss()

    generator.train()
    discriminator.train()

    total_g_loss = 0.0
    total_d_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        masked_input = batch["masked_input"].to(device)  # (N, B, H, W)
        target = batch["target"].to(device)              # (N, B, H, W)

        # -------------------------
        # Train Generator
        # -------------------------
        g_optimizer.zero_grad()

        generated = generator(masked_input)
        pred_fake = discriminator(masked_input, generated)

        valid = torch.ones_like(pred_fake, device=device)

        g_adv = adversarial_loss(pred_fake, valid)
        g_l1 = pixelwise_loss(generated, target)
        g_loss = g_adv + lambda_l1 * g_l1

        g_loss.backward()
        g_optimizer.step()

        # -------------------------
        # Train Discriminator
        # -------------------------
        d_optimizer.zero_grad()

        pred_real = discriminator(masked_input, target)
        valid = torch.ones_like(pred_real, device=device)
        loss_real = adversarial_loss(pred_real, valid)

        pred_fake_detached = discriminator(masked_input, generated.detach())
        fake = torch.zeros_like(pred_fake_detached, device=device)
        loss_fake = adversarial_loss(pred_fake_detached, fake)

        d_loss = 0.5 * (loss_real + loss_fake)

        d_loss.backward()
        d_optimizer.step()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()

    avg_g_loss = total_g_loss / len(dataloader)
    avg_d_loss = total_d_loss / len(dataloader)

    return avg_g_loss, avg_d_loss