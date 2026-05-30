import torch
import torch.nn as nn

from .layers import normalize_layers
from .config import NUM_CLASSES


class DynamicCNN(nn.Module):
    """CNN z warstwami konwolucyjnymi (4D) i głową klasyfikacji (2D po GAP)."""

    def __init__(self, layers: list[dict]):
        super().__init__()
        self.layers_cfg = normalize_layers(layers)
        spatial: list[nn.Module] = []
        dense: list[nn.Module] = []
        in_ch = 3
        feat_dim: int | None = None
        dense_mode = False
        flattened = False

        for layer in self.layers_cfg:
            t = layer["type"]

            if not dense_mode:
                if t == "conv":
                    out_ch = int(layer.get("filters", 32))
                    k = int(layer.get("kernel", 3))
                    spatial.append(nn.Conv2d(in_ch, out_ch, k, padding=k // 2))
                    in_ch = out_ch
                elif t == "batchnorm":
                    spatial.append(nn.BatchNorm2d(in_ch))
                elif t == "relu":
                    spatial.append(nn.ReLU(inplace=True))
                elif t == "maxpool":
                    spatial.append(nn.MaxPool2d(2, 2))
                elif t == "dropout":
                    spatial.append(nn.Dropout2d(float(layer.get("p", 0.2))))
                elif t == "globalaveragepooling":
                    spatial.append(nn.AdaptiveAvgPool2d(1))
                    spatial.append(nn.Flatten())
                    feat_dim = in_ch
                    dense_mode = True
                    flattened = True
                elif t == "linear":
                    # rzadki przypadek: spatial już 1x1 bez jawnego GAP
                    feat_dim = in_ch
                    dense_mode = True
                    dense.append(nn.Flatten())
                    flattened = True
                    units = int(layer.get("units", 128))
                    dense.append(nn.Linear(feat_dim, units))
                    dense.append(nn.ReLU(inplace=True))
                    feat_dim = units
                continue

            # głowa klasyfikacji (2D)
            if t == "linear":
                if not flattened:
                    dense.append(nn.Flatten())
                    flattened = True
                if feat_dim is None:
                    raise ValueError("Linear bez zdefiniowanego wymiaru cech")
                units = int(layer.get("units", 128))
                dense.append(nn.Linear(feat_dim, units))
                dense.append(nn.ReLU(inplace=True))
                feat_dim = units
            elif t == "batchnorm":
                if feat_dim is None:
                    raise ValueError("BatchNorm bez wymiaru cech")
                dense.append(nn.BatchNorm1d(feat_dim))
            elif t == "relu":
                dense.append(nn.ReLU(inplace=True))
            elif t == "dropout":
                dense.append(nn.Dropout(float(layer.get("p", 0.2))))
            elif t in ("maxpool", "conv", "globalaveragepooling"):
                raise ValueError(f"Warstwa '{t}' niedozwolona po GAP / w głowie klasyfikacji")

        if feat_dim is None:
            raise ValueError("Brak GAP lub warstwy Linear definiującej głowę")
        dense.append(nn.Linear(feat_dim, NUM_CLASSES))

        self.features = nn.Sequential(*spatial)
        self.classifier = nn.Sequential(*dense)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(arch: dict) -> nn.Module:
    return DynamicCNN(arch["layers"])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def smoke_test_forward(model: nn.Module, device: torch.device | None = None) -> None:
    """Szybki test kształtów – łapie błędy wymiarowe przed treningiem."""
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 32, 32, device=device)
        model(x)
