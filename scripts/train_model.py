import torch
from torch.utils.data import DataLoader

from src.dataset.dataset_loader import SentinelNPYDataset
from src.models.gan import build_models
from src.training.trainer import train_one_epoch


def main():
    print("START")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = SentinelNPYDataset(
        root_dir="data/raw/S2_Spectral_Bands",
        bands=["B01_SR"],
        rois=["roi1"],
        normalize="minus1_1",
        create_synthetic_mask=True,
        preload=False,
    )
    print("Dataset size:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    generator, discriminator = build_models(in_channels=1, out_channels=1)
    generator.to(device)
    discriminator.to(device)

    print("Models loaded")

    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=2e-4,
        betas=(0.5, 0.999)
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=2e-4,
        betas=(0.5, 0.999)
    )

    print("Start training...")

    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} starting...")
        g_loss, d_loss = train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            dataloader=dataloader,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            device=device,
        )
        print(f"Epoch [{epoch + 1}/{epochs}] | G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f}")

    print("DONE")


if __name__ == "__main__":
    main()