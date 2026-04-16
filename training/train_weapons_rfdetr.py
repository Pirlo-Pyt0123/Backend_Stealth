"""
Fine-tuning de RF-DETR Medium para detección de armas — LUCY.

Entrena el mismo modelo Transformer que ya usa el proyecto (RF-DETR)
sobre el dataset guns-knives-coco para especializar la detección de:
  - weapon   (arma genérica)
  - knife    (cuchillo)
  - pistol   (pistola)

Mantiene consistencia arquitectónica: todo Transformer, sin mezclar modelos.

Uso:
    cd c:/Users/LENOVO/Documents/Stealth
    python -m training.train_weapons_rfdetr
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json

DATASET_DIR = Path("guns-knives-coco/guns-knives-coco")
MODELS_DIR  = Path("models")


def verify_dataset():
    """Verifica que el dataset esté correcto antes de entrenar."""
    for split in ["train", "valid", "test"]:
        ann_file = DATASET_DIR / split / "_annotations.coco.json"
        if not ann_file.exists():
            print(f"  [ERROR] No encontrado: {ann_file}")
            return False
        with open(ann_file) as f:
            data = json.load(f)
        imgs = len(data["images"])
        anns = len(data["annotations"])
        cats = [c["name"] for c in data["categories"]]
        print(f"  {split:<6}: {imgs:4d} imgs  {anns:5d} anns  clases: {cats}")
    return True


def train():
    print("[LUCY] Fine-tuning RF-DETR para detección de armas")
    print(f"       Dataset: {DATASET_DIR}")
    print(f"       Modelo base: rf-detr-medium.pth\n")

    MODELS_DIR.mkdir(exist_ok=True)

    print("[1] Verificando dataset...")
    if not verify_dataset():
        print("  Dataset no encontrado. Verifica la ruta.")
        return

    print("\n[2] Cargando RF-DETR Medium...")
    try:
        from rfdetr import RFDETRMedium
    except ImportError:
        print("  [ERROR] rfdetr no instalado. Ejecuta: pip install rfdetr")
        return

    model = RFDETRMedium(pretrain_weights="rf-detr-medium.pth")

    print("\n[3] Iniciando fine-tuning...")
    print("    Epochs: 50  |  Batch: 8  |  LR: 1e-4")
    print("    Las curvas de entrenamiento se guardan en runs/weapons/\n")

    model.train(
        dataset_dir = str(DATASET_DIR),
        epochs      = 50,
        batch_size  = 8,
        lr          = 1e-4,
        output_dir  = "runs/weapons_rfdetr",
        # Callbacks internos de RF-DETR guardan métricas automáticamente
    )

    # Guardar modelo final
    out_path = MODELS_DIR / "weapon_rfdetr_finetuned.pth"
    import torch
    torch.save(model.state_dict(), out_path)
    print(f"\n[LUCY] Weapon RF-DETR guardado en: {out_path}")
    print(f"       Métricas en: runs/weapons_rfdetr/")


if __name__ == "__main__":
    train()
