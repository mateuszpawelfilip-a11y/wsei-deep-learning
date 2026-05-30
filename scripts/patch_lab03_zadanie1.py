import json
from pathlib import Path

nb_path = Path("WSEI_Lab_03_Generowanie_obrazków_z_pomocą_sieci_neuronowych.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

code = """# Zadanie 1: GAN (Generator + Dyskryminator) na MNIST i Fashion-MNIST
# %pip install torch torchvision matplotlib

import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab03.gan import GANConfig, Discriminator, Generator, plot_losses, show_sample_grids, train_gan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

cfg = GANConfig(
    latent_dim=100,
    batch_size=128,
    epochs=15,
    lr=2e-4,
    snapshot_epochs=(1, 5, 10, 15),
    num_workers=0,
)

OUT = Path("./outputs/lab03_zadanie1_gan")
histories = {}

for name in ("mnist", "fashion_mnist"):
    title = "MNIST" if name == "mnist" else "Fashion-MNIST"
    print(f"\\n{'='*50}\\nTrening GAN: {title}\\n{'='*50}")
    histories[title] = train_gan(
        dataset_name=name,
        out_dir=OUT / name,
        device=device,
        cfg=cfg,
        seed=42 if name == "mnist" else 123,
    )

plot_losses(histories, save_path=OUT / "losses.png")

show_sample_grids(
    {"MNIST": OUT / "mnist", "Fashion-MNIST": OUT / "fashion_mnist"},
    epochs=cfg.snapshot_epochs,
)

G = Generator(cfg.latent_dim)
D = Discriminator()
print("\\n--- Generator (G) ---")
print(G)
print(f"Parametry G: {sum(p.numel() for p in G.parameters()):,}")
print("\\n--- Dyskryminator (D) ---")
print(D)
print(f"Parametry D: {sum(p.numel() for p in D.parameters()):,}")
print(f"\\nWyniki zapisane w: {OUT.resolve()}")
"""

md = """### Analiza (Zadanie 1)

**Architektura:** DCGAN – G: ConvTranspose2d + BatchNorm + ReLU + Tanh; D: Conv2d + BatchNorm + LeakyReLU(0.2) + Sigmoid.

**Funkcje straty:** binarna entropia krzyżowa (BCE) – D uczy się klasyfikować real/fake; G maksymalizuje P(fake=real).

**Stabilność:** lekkie label smoothing (real=0.9, fake=0.1) ogranicza przeuczenie D. Obserwuj wykresy loss_D i loss_G.

**Jakość:** po ~10–15 epokach cyfry/ubrania są rozpoznawalne (MNIST zwykle szybciej). Epoki 1–5: szum; 10–15: ostrzejsze kształty.

**Porównanie:** MNIST – stabilniejszy trening; Fashion-MNIST – trudniejsze tekstury.

**Ograniczenia:** możliwy mode collapse przy zbyt silnym D lub złym LR.
"""

def _src(cell):
    return "".join(cell.get("source", []))

insert_at = None
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "markdown" and "Cel zadania 1" in _src(cell):
        insert_at = i + 1
        break

if insert_at is None:
    raise SystemExit("task 1 markdown not found")

if insert_at < len(nb["cells"]) and _src(nb["cells"][insert_at]).startswith("# Zadanie 1: GAN"):
    print("already patched")
else:
    code_cell = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [line + "\n" for line in code.split("\n")],
    }
    if code_cell["source"]:
        code_cell["source"][-1] = code_cell["source"][-1].rstrip("\n")
    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in md.split("\n")],
    }
    if md_cell["source"]:
        md_cell["source"][-1] = md_cell["source"][-1].rstrip("\n")
    nb["cells"].insert(insert_at, md_cell)
    nb["cells"].insert(insert_at, code_cell)

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok, cells:", len(nb["cells"]))
