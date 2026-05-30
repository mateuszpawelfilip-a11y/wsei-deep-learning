"""DCGAN dla obrazów 28x28 (1 kanał) – Zadanie 1 Lab 03."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


@dataclass
class GANConfig:
    latent_dim: int = 100
    batch_size: int = 128
    epochs: int = 15
    lr: float = 2e-4
    snapshot_epochs: tuple[int, ...] = (1, 5, 10, 15)
    num_workers: int = 0


@dataclass
class GANHistory:
    loss_d: list[float] = field(default_factory=list)
    loss_g: list[float] = field(default_factory=list)


class Generator(nn.Module):
    """DCGAN Generator: wektor Z -> obraz 1x28x28, aktywacja Tanh ([-1,1])."""

    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """DCGAN Discriminator: obraz -> P(prawdziwy), LeakyReLU + Sigmoid."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1)


def _make_loader(dataset_name: str, cfg: GANConfig) -> DataLoader:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    root = "./data"
    if dataset_name == "mnist":
        ds = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    elif dataset_name == "fashion_mnist":
        ds = datasets.FashionMNIST(root=root, train=True, download=True, transform=tfm)
    else:
        raise ValueError(dataset_name)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_gan(
    dataset_name: str,
    out_dir: str | Path,
    device: torch.device,
    cfg: GANConfig | None = None,
    seed: int = 42,
) -> GANHistory:
    cfg = cfg or GANConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    loader = _make_loader(dataset_name, cfg)

    G = Generator(cfg.latent_dim).to(device)
    D = Discriminator().to(device)
    criterion = nn.BCELoss()
    opt_g = optim.Adam(G.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=cfg.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, cfg.latent_dim, 1, 1, device=device)
    history = GANHistory()

    def _save_samples(epoch: int) -> None:
        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise).cpu()
        grid = utils.make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
        utils.save_image(grid, out_dir / f"samples_epoch_{epoch:03d}.png")

    for epoch in range(1, cfg.epochs + 1):
        G.train()
        D.train()
        epoch_loss_d = 0.0
        epoch_loss_g = 0.0
        n_batches = 0

        for real, _ in loader:
            real = real.to(device)
            bs = real.size(0)
            y_real = torch.ones(bs, 1, device=device) * 0.9
            y_fake = torch.zeros(bs, 1, device=device) + 0.1

            z = torch.randn(bs, cfg.latent_dim, 1, 1, device=device)
            fake = G(z).detach()

            loss_d = criterion(D(real), y_real) + criterion(D(fake), y_fake)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            z = torch.randn(bs, cfg.latent_dim, 1, 1, device=device)
            fake = G(z)
            loss_g = criterion(D(fake), torch.ones(bs, 1, device=device))
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            n_batches += 1

        epoch_loss_d /= max(n_batches, 1)
        epoch_loss_g /= max(n_batches, 1)
        history.loss_d.append(epoch_loss_d)
        history.loss_g.append(epoch_loss_g)

        if epoch in cfg.snapshot_epochs:
            _save_samples(epoch)

        print(
            f"[{dataset_name}] Epoch {epoch}/{cfg.epochs} | "
            f"loss_D={epoch_loss_d:.4f} | loss_G={epoch_loss_g:.4f}"
        )

    _save_samples(cfg.epochs)
    return history


def plot_losses(histories: dict[str, GANHistory], save_path: Path | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in histories.items():
        axes[0].plot(h.loss_d, label=name)
        axes[1].plot(h.loss_g, label=name)
    axes[0].set_title("Strata dyskryminatora D")
    axes[1].set_title("Strata generatora G")
    for ax in axes:
        ax.set_xlabel("Epoka")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def show_sample_grids(out_dirs: dict[str, Path], epochs: tuple[int, ...]) -> plt.Figure:
    n = len(epochs)
    fig, axes = plt.subplots(len(out_dirs), n, figsize=(3 * n, 3 * len(out_dirs)))
    if len(out_dirs) == 1:
        axes = axes.reshape(1, -1)
    for row, (title, odir) in enumerate(out_dirs.items()):
        for col, ep in enumerate(epochs):
            path = Path(odir) / f"samples_epoch_{ep:03d}.png"
            if not path.exists():
                raise FileNotFoundError(f"Brak pliku: {path} – uruchom trening GAN.")
            img = plt.imread(path)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"epoka {ep}")
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=11)
    plt.suptitle("Wygenerowane próbki na różnych etapach treningu", y=1.02)
    plt.tight_layout()
    return fig


def display_results_in_notebook(
    out_base: Path,
    histories: dict[str, GANHistory] | None = None,
    snapshot_epochs: tuple[int, ...] = (1, 5, 10, 15),
) -> None:
    """Wyświetl wykresy i PNG w notebooku (Jupyter / VS Code)."""
    try:
        from IPython.display import Image as IPImage, display
    except ImportError:
        return

    out_base = Path(out_base)
    losses_png = out_base / "losses.png"

    if histories:
        fig = plot_losses(histories, save_path=losses_png)
        display(fig)
        plt.close(fig)
    elif losses_png.exists():
        display(IPImage(filename=str(losses_png)))

    fig = show_sample_grids(
        {"MNIST": out_base / "mnist", "Fashion-MNIST": out_base / "fashion_mnist"},
        epochs=snapshot_epochs,
    )
    display(fig)
    plt.close(fig)

    for ds in ("mnist", "fashion_mnist"):
        title = "MNIST" if ds == "mnist" else "Fashion-MNIST"
        final = out_base / ds / f"samples_epoch_{max(snapshot_epochs):03d}.png"
        if final.exists():
            print(f"\n{title} – próbki końcowe ({final.name}):")
            display(IPImage(filename=str(final), width=500))
