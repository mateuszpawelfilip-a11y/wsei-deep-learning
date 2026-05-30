import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import BATCH_SIZE, NUM_CLASSES, PROJECT_DIR, TRAIN_EPOCHS


def get_cifar_loaders(batch_size: int = BATCH_SIZE):
    train_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    root = PROJECT_DIR / "data"
    train_ds = datasets.CIFAR10(root, train=True, download=True, transform=train_t)
    test_ds = datasets.CIFAR10(root, train=False, download=True, transform=test_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


def train_architecture(
    model: nn.Module,
    device: torch.device,
    epochs: int = TRAIN_EPOCHS,
    train_loader=None,
    test_loader=None,
) -> float:
    """Trenuje model i zwraca accuracy na zbiorze testowym (0–1)."""
    if train_loader is None or test_loader is None:
        train_loader, test_loader = get_cifar_loaders()

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total
