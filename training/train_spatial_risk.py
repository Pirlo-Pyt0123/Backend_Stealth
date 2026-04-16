"""
Entrenamiento del SpatialRiskTransformer de LUCY.

Aprende a evaluar riesgo vial desde la perspectiva del PEATÓN.
Sin distancias hardcodeadas. El Transformer descubre por sí mismo
qué configuraciones espaciales son peligrosas.

Datos:
  - Escenarios sintéticos ricos basados en COCO class IDs
  - Peatón + vehículo en múltiples configuraciones
  - Datasets descargados: pedestrian_detection, vehicle_detection, jaad_annotations

El modelo aprende que:
  - Peatón + vehículo cerca = WARNING o CRITICAL
  - Peatón en calzada sin cruce = WARNING
  - Zona oscura + tráfico = CRITICAL
  - Ciclista entre vehículos grandes = WARNING
  etc. (sin que nadie se lo diga explícitamente)

Uso:
    cd c:/Users/LENOVO/Documents/Stealth
    python -m training.train_spatial_risk
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
from pathlib import Path

from models.spatial_risk_net import (
    SpatialRiskTransformer, RISK_CLASSES, N_RISK,
    MAX_OBJECTS, N_COCO_CLASSES, detections_to_tokens
)

# ── Configuración ─────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS     = 50
LR         = 2e-4
MODELS_DIR = Path("models")
RNG        = np.random.default_rng(42)

# COCO IDs
PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK = 0, 1, 2, 3, 5, 7
TRAFFIC_LIGHT, STOP_SIGN = 9, 11
LARGE_VEHICLES = {BUS, TRUCK}
ALL_VEHICLES   = {BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK}

print(f"[LUCY] Entrenando SpatialRiskTransformer en: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR DE ESCENARIOS
# Cada muestra = lista de detecciones con class_id, bbox, confidence
# ─────────────────────────────────────────────────────────────────────────────

def _det(cls_id, cx, cy, w, h, conf=None, jitter=0.03):
    """Crea detección con jitter realista."""
    cx = float(np.clip(cx + RNG.normal(0, jitter), 0.02, 0.98))
    cy = float(np.clip(cy + RNG.normal(0, jitter), 0.02, 0.98))
    w  = float(np.clip(w  * RNG.uniform(0.85, 1.15), 0.01, 0.6))
    h  = float(np.clip(h  * RNG.uniform(0.85, 1.15), 0.01, 0.6))
    conf = conf or float(RNG.uniform(0.55, 0.99))
    x1 = max(0.0, cx - w/2); y1 = max(0.0, cy - h/2)
    x2 = min(1.0, cx + w/2); y2 = min(1.0, cy + h/2)
    return {"class_id": cls_id, "bbox": [x1*640, y1*640, x2*640, y2*640], "confidence": conf}


def _rand():
    return float(RNG.uniform(0.1, 0.9))


# ── SAFE scenarios ────────────────────────────────────────────────────────────
def safe_empty_street(n=1500):
    samples = []
    for _ in range(n):
        dets = []
        np_ = int(RNG.integers(0, 3))
        for _ in range(np_):
            dets.append(_det(PERSON, _rand(), _rand(), 0.05, 0.15))
        samples.append((dets, 0))
    return samples

def safe_sidewalk_only(n=1500):
    """Personas en vereda, sin vehículos cercanos."""
    samples = []
    for _ in range(n):
        dets = [_det(PERSON, float(RNG.uniform(0.05,0.25)), _rand(), 0.05, 0.15)
                for _ in range(int(RNG.integers(1,4)))]
        if RNG.random() > 0.5:
            dets.append(_det(CAR, float(RNG.uniform(0.6,0.95)), _rand(), 0.20, 0.12, jitter=0.01))
        samples.append((dets, 0))
    return samples


# ── WARNING scenarios ─────────────────────────────────────────────────────────
def warning_pedestrian_near_traffic(n=1500):
    """Peatón cerca de vehículos (no overlap, pero peligroso)."""
    samples = []
    for _ in range(n):
        vx = float(RNG.uniform(0.3, 0.7))
        vy = float(RNG.uniform(0.3, 0.7))
        offset = float(RNG.uniform(0.10, 0.22))
        side   = float(RNG.choice([-1,1]))
        px = float(np.clip(vx + side*offset, 0.05, 0.95))
        dets = [
            _det(PERSON, px, vy, 0.05, 0.15),
            _det(int(RNG.choice([CAR, MOTORCYCLE])), vx, vy, 0.18, 0.11),
        ]
        for _ in range(int(RNG.integers(0,3))):
            dets.append(_det(int(RNG.choice([CAR, MOTORCYCLE])), _rand(), _rand(), 0.18,0.11))
        samples.append((dets, 1))
    return samples

def warning_pedestrian_on_road(n=1500):
    """Peatón caminando por calzada (zona central)."""
    samples = []
    for _ in range(n):
        px = float(RNG.uniform(0.35, 0.65))
        dets = [_det(PERSON, px, float(RNG.uniform(0.3,0.7)), 0.05, 0.15)]
        for _ in range(int(RNG.integers(1,4))):
            dets.append(_det(int(RNG.choice([CAR,BUS,TRUCK])), _rand(), _rand(), 0.20,0.12))
        samples.append((dets, 1))
    return samples

def warning_cyclist_exposed(n=1500):
    """Ciclista entre vehículos."""
    samples = []
    for _ in range(n):
        bx, by = _rand(), _rand()
        dets = [_det(BICYCLE, bx, by, 0.07, 0.12)]
        # Vehículo cercano
        offset = float(RNG.uniform(0.12, 0.25))
        dets.append(_det(int(RNG.choice([CAR, BUS, TRUCK])),
                         float(np.clip(bx + RNG.choice([-1,1])*offset,0.05,0.95)),
                         by, 0.22, 0.14))
        for _ in range(int(RNG.integers(0,3))):
            dets.append(_det(int(RNG.choice([CAR,TRUCK])), _rand(), _rand(), 0.20,0.12))
        samples.append((dets, 1))
    return samples

def warning_dark_zone(n=1000):
    """Peatón en zona oscura con tráfico (señalado con vehículos y persona)."""
    samples = []
    for _ in range(n):
        dets = [_det(PERSON, _rand(), _rand(), 0.05, 0.15, conf=float(RNG.uniform(0.4,0.7)))]
        for _ in range(int(RNG.integers(1,3))):
            dets.append(_det(int(RNG.choice([CAR,MOTORCYCLE])), _rand(), _rand(), 0.18,0.11,
                              conf=float(RNG.uniform(0.4,0.7))))
        samples.append((dets, 1))
    return samples

def warning_phone_crossing(n=1000):
    """Peatón cruzando con confianza baja (distracción simulada) cerca de tráfico."""
    samples = []
    for _ in range(n):
        cx = float(RNG.uniform(0.35,0.65))
        dets = [_det(PERSON, cx, 0.5, 0.05, 0.15, conf=float(RNG.uniform(0.45,0.70)))]
        for _ in range(int(RNG.integers(1,3))):
            vx = float(np.clip(cx + RNG.choice([-1,1])*float(RNG.uniform(0.1,0.25)),0.05,0.95))
            dets.append(_det(CAR, vx, 0.5, 0.20, 0.12))
        samples.append((dets, 1))
    return samples


# ── CRITICAL scenarios ────────────────────────────────────────────────────────
def critical_imminent_collision(n=1500):
    """Peatón con overlap real sobre vehículo."""
    samples = []
    for _ in range(n):
        cx, cy = _rand(), _rand()
        dets = [
            _det(PERSON, cx, cy, 0.06, 0.16, jitter=0.005),
            _det(int(RNG.choice([CAR, MOTORCYCLE, BUS, TRUCK])),
                 cx + float(RNG.normal(0, 0.025)),
                 cy + float(RNG.normal(0, 0.025)),
                 0.22, 0.14, jitter=0.005),
        ]
        for _ in range(int(RNG.integers(0,3))):
            dets.append(_det(int(RNG.choice([CAR,TRUCK])), _rand(), _rand(), 0.20,0.12))
        samples.append((dets, 2))
    return samples

def critical_large_vehicle_blind_spot(n=1500):
    """Peatón/ciclista en punto ciego de camión o bus."""
    samples = []
    for _ in range(n):
        lx, ly = _rand(), _rand()
        large = _det(int(RNG.choice([BUS, TRUCK])), lx, ly, 0.30, 0.18, jitter=0.01)
        offset = float(RNG.uniform(0.02, 0.09))
        side   = float(RNG.choice([-1, 1]))
        small_cls = int(RNG.choice([PERSON, BICYCLE]))
        small = _det(small_cls,
                     float(np.clip(lx+side*offset,0.02,0.98)),
                     float(np.clip(ly+RNG.uniform(-0.05,0.05),0.02,0.98)),
                     0.05, 0.14, jitter=0.01)
        dets = [large, small]
        samples.append((dets, 2))
    return samples

def critical_child_near_traffic(n=1000):
    """Persona pequeña (niño sim.) sola entre vehículos grandes."""
    samples = []
    for _ in range(n):
        cx = float(RNG.uniform(0.3,0.7))
        dets = [_det(PERSON, cx, 0.5, 0.03, 0.08)]  # bbox pequeño = niño
        for _ in range(int(RNG.integers(2,5))):
            dets.append(_det(int(RNG.choice([CAR,BUS,TRUCK])), _rand(), _rand(), 0.22,0.13))
        samples.append((dets, 2))
    return samples

def critical_running_into_traffic(n=1000):
    """Múltiples vehículos + peatón en posición central (corriendo)."""
    samples = []
    for _ in range(n):
        dets = [_det(PERSON, 0.5, 0.5, 0.05, 0.15, conf=float(RNG.uniform(0.7,0.99)))]
        for _ in range(int(RNG.integers(3,6))):
            dets.append(_det(int(RNG.choice([CAR,BUS,TRUCK,MOTORCYCLE])),
                              _rand(), _rand(), 0.20,0.12))
        samples.append((dets, 2))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
def build_dataset():
    print("\n[1] Generando escenarios de entrenamiento...")
    all_samples = []

    generators = [
        # SAFE
        ("safe_empty",         safe_empty_street,           1500),
        ("safe_sidewalk",      safe_sidewalk_only,          1500),
        # WARNING
        ("warn_near_traffic",  warning_pedestrian_near_traffic, 1500),
        ("warn_on_road",       warning_pedestrian_on_road,   1500),
        ("warn_cyclist",       warning_cyclist_exposed,      1500),
        ("warn_dark",          warning_dark_zone,            1000),
        ("warn_phone",         warning_phone_crossing,       1000),
        # CRITICAL
        ("crit_collision",     critical_imminent_collision,  1500),
        ("crit_blind_spot",    critical_large_vehicle_blind_spot, 1500),
        ("crit_child",         critical_child_near_traffic,  1000),
        ("crit_running",       critical_running_into_traffic,1000),
    ]

    for name, fn, n in generators:
        data = fn(n)
        all_samples.extend(data)
        labels = [lbl for _, lbl in data]
        counts = np.bincount(labels, minlength=3)
        print(f"   {name:<25} {len(data):5d}  →  {RISK_CLASSES[labels[0]]}")

    labels_all = [lbl for _, lbl in all_samples]
    print(f"\n   Total: {len(all_samples)} escenarios")
    for i, cls in enumerate(RISK_CLASSES):
        print(f"   {cls:<10}: {labels_all.count(i):5d}")

    return all_samples


class RiskDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dets, label = self.samples[idx]
        tokens, mask = detections_to_tokens(dets, img_w=640, img_h=640)
        return tokens, mask, torch.tensor(label, dtype=torch.long)


def train():
    all_samples = build_dataset()
    labels = [lbl for _, lbl in all_samples]
    idx_tr, idx_val = train_test_split(
        range(len(all_samples)), test_size=0.15, random_state=42,
        stratify=labels
    )
    train_data = [all_samples[i] for i in idx_tr]
    val_data   = [all_samples[i] for i in idx_val]

    train_labels = [lbl for _, lbl in train_data]
    counts   = np.bincount(train_labels)
    weights  = 1.0 / counts[train_labels]
    sampler  = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_ds = RiskDataset(train_data)
    val_ds   = RiskDataset(val_data)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

    model     = SpatialRiskTransformer().to(DEVICE)
    # Pesos de clase: CRITICAL tiene mayor peso (no podemos perderlo)
    class_weights = torch.tensor([1.0, 1.5, 2.5], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": [],
               "precision": [], "recall": []}

    print(f"\n[2] Entrenando SpatialRiskTransformer — {EPOCHS} epochs en {DEVICE}")
    print(f"    Parámetros: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for tokens, mask, labels_b in train_dl:
            tokens  = tokens.to(DEVICE)
            mask    = mask.to(DEVICE)
            labels_b= labels_b.to(DEVICE)
            out     = model(tokens, mask)
            loss    = criterion(out["logits"], labels_b)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        model.eval()
        v_loss, all_preds, all_true = 0.0, [], []
        with torch.no_grad():
            for tokens, mask, labels_b in val_dl:
                tokens   = tokens.to(DEVICE)
                mask     = mask.to(DEVICE)
                labels_b = labels_b.to(DEVICE)
                out      = model(tokens, mask)
                v_loss  += criterion(out["logits"], labels_b).item()
                all_preds.extend(out["probs"].argmax(1).cpu().tolist())
                all_true.extend(labels_b.cpu().tolist())

        acc = sum(p==t for p,t in zip(all_preds, all_true)) / len(all_true) * 100
        history["train_loss"].append(t_loss / len(train_dl))
        history["val_loss"].append(v_loss / len(val_dl))
        history["val_acc"].append(acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"train={history['train_loss'][-1]:.4f}  "
                  f"val={history['val_loss'][-1]:.4f}  "
                  f"acc={acc:.1f}%")

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), MODELS_DIR / "spatial_risk_best.pth")

    # Reporte final
    model.load_state_dict(torch.load(MODELS_DIR / "spatial_risk_best.pth"))
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for tokens, mask, labels_b in val_dl:
            out = model(tokens.to(DEVICE), mask.to(DEVICE))
            all_preds.extend(out["probs"].argmax(1).cpu().tolist())
            all_true.extend(labels_b.tolist())

    print(f"\n[3] Reporte final (val_acc={best_val_acc:.1f}%)")
    print(classification_report(all_true, all_preds,
                                  target_names=RISK_CLASSES, zero_division=0))

    with open(MODELS_DIR / "spatial_risk_history.pkl", "wb") as f:
        pickle.dump(history, f)

    model.export_onnx(str(MODELS_DIR / "spatial_risk.onnx"))
    print(f"\n[LUCY] SpatialRiskTransformer entrenado. Mejor acc: {best_val_acc:.1f}%")
    return model, history


if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    train()
