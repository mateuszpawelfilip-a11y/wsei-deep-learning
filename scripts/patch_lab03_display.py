import json
from pathlib import Path

nb_path = Path("WSEI_Lab_03_Generowanie_obrazków_z_pomocą_sieci_neuronowych.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

train_code = """# Zadanie 1: GAN – trening (MNIST + Fashion-MNIST)
# %pip install torch torchvision matplotlib

%matplotlib inline

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab03.gan import GANConfig, Discriminator, Generator, display_results_in_notebook, train_gan

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

G = Generator(cfg.latent_dim)
D = Discriminator()
print("\\n--- Generator (G) ---")
print(G)
print(f"Parametry G: {sum(p.numel() for p in G.parameters()):,}")
print("\\n--- Dyskryminator (D) ---")
print(D)
print(f"Parametry D: {sum(p.numel() for p in D.parameters()):,}")
print(f"\\nPliki zapisane w: {OUT.resolve()}")

# Wyświetl wykresy i obrazki bezpośrednio w notebooku
display_results_in_notebook(OUT, histories=histories, snapshot_epochs=cfg.snapshot_epochs)
"""

display_code = """# Zadanie 1: podgląd wyników w notebooku (bez ponownego treningu)
%matplotlib inline

from pathlib import Path
from lab03.gan import GANConfig, display_results_in_notebook

OUT = Path("./outputs/lab03_zadanie1_gan")
cfg = GANConfig()

if not (OUT / "mnist").exists():
    raise FileNotFoundError(
        f"Brak wyników w {OUT}. Najpierw uruchom komórkę treningu powyżej."
    )

display_results_in_notebook(OUT, histories=None, snapshot_epochs=cfg.snapshot_epochs)
print("Gotowe – wykresy i siatki powyżej.")
"""

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if cell.get("cell_type") == "code" and src.startswith("# Zadanie 1: GAN"):
        cell["source"] = [line + "\n" for line in train_code.split("\n")]
        if cell["source"]:
            cell["source"][-1] = cell["source"][-1].rstrip("\n")
        # insert display cell after train if not present
        next_src = "".join(nb["cells"][i + 1].get("source", [])) if i + 1 < len(nb["cells"]) else ""
        if not next_src.startswith("# Zadanie 1: podgląd"):
            disp = {
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
                "source": [line + "\n" for line in display_code.split("\n")],
            }
            if disp["source"]:
                disp["source"][-1] = disp["source"][-1].rstrip("\n")
            nb["cells"].insert(i + 1, disp)
        break
else:
    raise SystemExit("train cell not found")

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("updated")
