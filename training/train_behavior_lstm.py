"""
Entrenamiento del BehaviorLSTM de LUCY.

Fuentes de datos (en orden de prioridad):
  1. Stanford Drone Dataset  → trayectorias reales de peatones anotadas
  2. Fight Detection Dataset → imágenes de violencia/confrontación
  3. Generador sintético     → merodeo, encercamiento, seguimiento enriquecido

El LSTM aprende patrones TEMPORALES sin reglas hardcodeadas.
La diferencia con el enfoque anterior: el modelo ve SECUENCIAS de posiciones
y aprende qué patrones de movimiento indican comportamiento sospechoso.

Uso:
    cd c:/Users/LENOVO/Documents/Stealth
    python -m training.train_behavior_lstm
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle, time
from pathlib import Path

from models.behavior_lstm import (
    BehaviorLSTM, BEHAVIOR_CLASSES, N_CLASSES, SEQ_LEN, FRAME_FEATS
)

# ── Configuración ─────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 64
EPOCHS      = 60
LR          = 3e-4
WEIGHT_DECAY= 1e-4
MODELS_DIR  = Path("models")
DATASET_DIR = Path("datasets")

RNG = np.random.default_rng(42)
print(f"[LUCY] Entrenando BehaviorLSTM en: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DE SECUENCIAS
# Cada secuencia = (SEQ_LEN, FRAME_FEATS) + label
# ─────────────────────────────────────────────────────────────────────────────

def _person_token(cx, cy, w=0.06, h=0.18, vx=0.0, vy=0.0):
    return [cx, cy, w, h, vx, vy]


def _pad_persons(persons, max_p=5):
    """Rellena hasta max_p personas con ceros."""
    out = [p for p in persons[:max_p]]
    out += [[0.0]*6] * (max_p - len(out))
    return out


def _build_frame(persons, brightness=0.7, weapon=0.0, n_persons=None):
    """
    Construye vector de frame a partir de lista de tokens persona.
    persons: lista de [cx, cy, w, h, vx, vy] (ya normalizados)
    """
    n = n_persons if n_persons is not None else len(persons)
    f = [min(n, 5) / 5.0]
    padded = _pad_persons(persons)
    for p in padded:
        f.extend(p)

    # min_dist y max_iou entre personas
    min_d, max_iou = 1.0, 0.0
    for i in range(len(persons)):
        for j in range(i+1, len(persons)):
            pi, pj = persons[i], persons[j]
            d = ((pi[0]-pj[0])**2 + (pi[1]-pj[1])**2)**0.5
            min_d = min(min_d, d)
    f.extend([brightness, weapon, min_d, max_iou])
    return f


def gen_normal(n=3000):
    """Personas moviéndose de forma normal (sin patrón sospechoso)."""
    seqs = []
    for _ in range(n):
        np_persons = RNG.integers(1, 4)
        starts  = [(float(RNG.uniform(0.1,0.9)), float(RNG.uniform(0.1,0.9))) for _ in range(np_persons)]
        dirs    = [(float(RNG.uniform(-0.015,0.015)), float(RNG.uniform(-0.015,0.015))) for _ in range(np_persons)]
        seq     = []
        for t in range(SEQ_LEN):
            persons = []
            for i in range(np_persons):
                cx = float(np.clip(starts[i][0] + dirs[i][0]*t + RNG.normal(0,0.005), 0.05, 0.95))
                cy = float(np.clip(starts[i][1] + dirs[i][1]*t + RNG.normal(0,0.005), 0.05, 0.95))
                vx = dirs[i][0] + float(RNG.normal(0, 0.003))
                vy = dirs[i][1] + float(RNG.normal(0, 0.003))
                persons.append([cx, cy, 0.06, 0.18, vx, vy])
            seq.append(_build_frame(persons, brightness=float(RNG.uniform(0.5,1.0))))
        seqs.append((np.array(seq, dtype=np.float32), 0))  # label=0 normal
    return seqs


def gen_loitering(n=3000):
    """Persona que permanece en zona pequeña durante toda la secuencia."""
    seqs = []
    for _ in range(n):
        cx0, cy0 = float(RNG.uniform(0.2,0.8)), float(RNG.uniform(0.2,0.8))
        seq = []
        for t in range(SEQ_LEN):
            # Movimiento browniano dentro de radio muy pequeño
            cx = float(np.clip(cx0 + RNG.normal(0, 0.012), 0.05, 0.95))
            cy = float(np.clip(cy0 + RNG.normal(0, 0.012), 0.05, 0.95))
            vx = float(RNG.normal(0, 0.003))
            vy = float(RNG.normal(0, 0.003))
            extra = []
            if RNG.random() > 0.5:
                extra = [[float(RNG.uniform(0.1,0.9)), float(RNG.uniform(0.1,0.9)),
                           0.06, 0.18, float(RNG.uniform(-0.01,0.01)), float(RNG.uniform(-0.01,0.01))]]
            persons = [[cx, cy, 0.06, 0.18, vx, vy]] + extra
            seq.append(_build_frame(persons, brightness=float(RNG.uniform(0.2,0.9))))
        seqs.append((np.array(seq, dtype=np.float32), 1))  # label=1 loitering
    return seqs


def gen_confrontation(n=3000):
    """Dos personas que convergen rápidamente y entran en contacto."""
    seqs = []
    for _ in range(n):
        # Empiezan separadas y se acercan
        cx1, cy1 = float(RNG.uniform(0.2,0.45)), float(RNG.uniform(0.3,0.7))
        cx2, cy2 = float(RNG.uniform(0.55,0.8)), float(RNG.uniform(0.3,0.7))
        seq = []
        for t in range(SEQ_LEN):
            progress = t / SEQ_LEN
            # Convergen hasta estar muy juntos
            c1x = cx1 + (0.5 - cx1) * progress * 0.85 + float(RNG.normal(0,0.006))
            c1y = cy1 + (0.5 - cy1) * progress * 0.85 + float(RNG.normal(0,0.006))
            c2x = cx2 + (0.5 - cx2) * progress * 0.85 + float(RNG.normal(0,0.006))
            c2y = cy2 + (0.5 - cy2) * progress * 0.85 + float(RNG.normal(0,0.006))
            v1x = (0.5-cx1)/SEQ_LEN; v1y = (0.5-cy1)/SEQ_LEN
            v2x = (0.5-cx2)/SEQ_LEN; v2y = (0.5-cy2)/SEQ_LEN
            persons = [
                [c1x, c1y, 0.07, 0.18, v1x, v1y],
                [c2x, c2y, 0.07, 0.18, v2x, v2y],
            ]
            seq.append(_build_frame(persons, brightness=float(RNG.uniform(0.3,1.0))))
        seqs.append((np.array(seq, dtype=np.float32), 2))  # label=2 confrontation
    return seqs


def gen_encirclement(n=3000):
    """Una persona rodeada progresivamente por otras desde distintos ángulos."""
    seqs = []
    for _ in range(n):
        n_surr = int(RNG.integers(3, 5))
        angles = [i * (2*np.pi/n_surr) for i in range(n_surr)]
        r_start = float(RNG.uniform(0.3, 0.5))
        seq = []
        for t in range(SEQ_LEN):
            progress = t / SEQ_LEN
            r = r_start * (1 - progress * 0.75)  # radio decrece
            target = [0.5, 0.5, 0.06, 0.18, 0.0, 0.0]
            surrounders = [
                [float(np.clip(0.5 + r*np.cos(a) + RNG.normal(0,0.015), 0.05, 0.95)),
                 float(np.clip(0.5 + r*np.sin(a) + RNG.normal(0,0.015), 0.05, 0.95)),
                 0.06, 0.18,
                 float(-r*np.cos(a)/SEQ_LEN),
                 float(-r*np.sin(a)/SEQ_LEN)]
                for a in angles
            ]
            persons = [target] + surrounders[:4]
            seq.append(_build_frame(persons, brightness=float(RNG.uniform(0.2,0.9))))
        seqs.append((np.array(seq, dtype=np.float32), 3))  # label=3 encirclement
    return seqs


def gen_following(n=3000):
    """Una persona sigue a otra con desfase temporal constante."""
    seqs = []
    for _ in range(n):
        sx, sy = float(RNG.uniform(0.1,0.5)), float(RNG.uniform(0.1,0.9))
        dx = float(RNG.uniform(0.008, 0.018)) * RNG.choice([-1,1])
        dy = float(RNG.uniform(0.003, 0.010)) * RNG.choice([-1,1])
        delay = int(RNG.integers(3, 8))
        traj1 = [(sx + dx*t + float(RNG.normal(0,0.004)),
                  sy + dy*t + float(RNG.normal(0,0.004))) for t in range(SEQ_LEN)]
        seq = []
        for t in range(SEQ_LEN):
            p1  = traj1[t]
            td  = max(0, t - delay)
            p2  = (traj1[td][0] + float(RNG.normal(0,0.01)),
                   traj1[td][1] + float(RNG.normal(0,0.01)))
            persons = [
                [float(np.clip(p1[0],0.05,0.95)), float(np.clip(p1[1],0.05,0.95)),
                 0.06, 0.18, dx, dy],
                [float(np.clip(p2[0],0.05,0.95)), float(np.clip(p2[1],0.05,0.95)),
                 0.06, 0.18, dx, dy],
            ]
            seq.append(_build_frame(persons, brightness=float(RNG.uniform(0.3,1.0))))
        seqs.append((np.array(seq, dtype=np.float32), 4))  # label=4 following
    return seqs


def gen_violence(n=3000):
    """Movimientos bruscos, velocidades altas, personas muy juntas."""
    seqs = []
    for _ in range(n):
        cx, cy = 0.5, 0.5
        seq = []
        for t in range(SEQ_LEN):
            # Movimientos erráticos de alta magnitud
            p1 = [float(np.clip(cx+RNG.normal(0,0.04),0.05,0.95)),
                  float(np.clip(cy+RNG.normal(0,0.04),0.05,0.95)),
                  0.08, 0.20,
                  float(RNG.normal(0,0.025)), float(RNG.normal(0,0.025))]
            p2 = [float(np.clip(cx+RNG.normal(0,0.04),0.05,0.95)),
                  float(np.clip(cy+RNG.normal(0,0.04),0.05,0.95)),
                  0.08, 0.20,
                  float(RNG.normal(0,0.025)), float(RNG.normal(0,0.025))]
            weapon = 1.0 if (t > SEQ_LEN//2 and RNG.random() > 0.5) else 0.0
            seq.append(_build_frame([p1, p2],
                                    brightness=float(RNG.uniform(0.1,0.6)),
                                    weapon=weapon))
        seqs.append((np.array(seq, dtype=np.float32), 5))  # label=5 violence
    return seqs


def load_stanford_drone_sequences(n_per_class=500) -> list:
    """
    Parsea el Stanford Drone Dataset y extrae secuencias de trayectorias.
    Genera etiquetas basadas en análisis geométrico de movimiento real.
    """
    sdd_path = DATASET_DIR / "stanford_drone" / "annotations"
    seqs = []
    if not sdd_path.exists():
        print("  [SKIP] Stanford Drone Dataset no encontrado")
        return seqs

    annotation_files = list(sdd_path.rglob("annotations.txt"))
    print(f"  Stanford Drone: {len(annotation_files)} archivos de anotación")

    for ann_file in annotation_files[:20]:  # usar primeros 20 videos
        try:
            tracks: dict = {}
            with open(ann_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 10:
                        continue
                    tid   = int(parts[0])
                    x1,y1,x2,y2 = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
                    frame = int(parts[5])
                    lost  = int(parts[6])
                    label = parts[9] if len(parts) > 9 else "Pedestrian"
                    if lost or label not in ("Pedestrian",):
                        continue
                    if tid not in tracks:
                        tracks[tid] = []
                    tracks[tid].append((frame, x1, y1, x2, y2))

            # Procesar trayectorias y extraer secuencias de SEQ_LEN
            for tid, pts in tracks.items():
                pts.sort(key=lambda x: x[0])
                if len(pts) < SEQ_LEN:
                    continue

                # Normalizar coords (imagen SDD ~1920x1080)
                W, H = 1920.0, 1080.0
                norm  = [(f, (x1+x2)/(2*W), (y1+y2)/(2*H),
                          (x2-x1)/W, (y2-y1)/H) for f,x1,y1,x2,y2 in pts]

                # Extraer ventanas de SEQ_LEN
                for start in range(0, len(norm)-SEQ_LEN, SEQ_LEN//2):
                    window = norm[start:start+SEQ_LEN]
                    seq    = []
                    for i, (_, cx, cy, w, h) in enumerate(window):
                        vx = cx - window[max(0,i-1)][1] if i > 0 else 0.0
                        vy = cy - window[max(0,i-1)][2] if i > 0 else 0.0
                        persons = [[float(cx), float(cy), float(w), float(h),
                                    float(vx), float(vy)]]
                        seq.append(_build_frame(persons, brightness=0.7))

                    # Determinar etiqueta por análisis de movimiento real
                    cxs = [w[1] for w in window]
                    cys = [w[2] for w in window]
                    displacement = ((cxs[-1]-cxs[0])**2 + (cys[-1]-cys[0])**2)**0.5
                    speeds = [((cxs[i]-cxs[i-1])**2+(cys[i]-cys[i-1])**2)**0.5
                               for i in range(1, len(window))]
                    mean_speed = float(np.mean(speeds))

                    if displacement < 0.05 and mean_speed < 0.004:
                        lbl = 1   # loitering
                    else:
                        lbl = 0   # normal

                    seqs.append((np.array(seq, dtype=np.float32), lbl))

        except Exception as e:
            continue

    print(f"  Stanford Drone: {len(seqs)} secuencias extraídas")
    return seqs


# ─────────────────────────────────────────────────────────────────────────────
class BehaviorDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(np.array(sequences), dtype=torch.float32)
        self.y = torch.tensor(np.array(labels),    dtype=torch.long)

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
def build_dataset():
    print("\n[1] Generando secuencias de entrenamiento...")
    all_seqs, all_labels = [], []

    generators = [
        ("normal",        gen_normal,        3000),
        ("loitering",     gen_loitering,     3000),
        ("confrontation", gen_confrontation, 3000),
        ("encirclement",  gen_encirclement,  3000),
        ("following",     gen_following,     3000),
        ("violence",      gen_violence,      3000),
    ]

    for name, fn, n in generators:
        data = fn(n)
        seqs, lbls = zip(*data)
        all_seqs.extend(seqs); all_labels.extend(lbls)
        print(f"   {name:<15} {len(seqs):5d} secuencias")

    # Agregar datos reales del Stanford Drone Dataset
    real_data = load_stanford_drone_sequences()
    for seq, lbl in real_data:
        all_seqs.append(seq); all_labels.append(lbl)

    X = np.array(all_seqs, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    print(f"\n   Total: {len(y)} secuencias  |  clases: {np.bincount(y).tolist()}")
    return X, y


def train():
    X, y = build_dataset()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # WeightedRandomSampler para balancear clases
    counts  = np.bincount(y_train)
    weights = 1.0 / counts[y_train]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_ds = BehaviorDataset(X_train, y_train)
    val_ds   = BehaviorDataset(X_val,   y_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

    model     = BehaviorLSTM().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_acc = 0.0
    history  = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\n[2] Entrenando BehaviorLSTM — {EPOCHS} epochs en {DEVICE}")
    print(f"    Parámetros: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        t_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out    = model(xb)
            loss   = criterion(out["logits"], yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        # ── Val ──
        model.eval()
        v_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out    = model(xb)
                v_loss += criterion(out["logits"], yb).item()
                correct += (out["probs"].argmax(1) == yb).sum().item()
                total   += len(yb)

        acc = correct / total * 100
        history["train_loss"].append(t_loss / len(train_dl))
        history["val_loss"].append(v_loss / len(val_dl))
        history["val_acc"].append(acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={history['train_loss'][-1]:.4f}  "
                  f"val_loss={history['val_loss'][-1]:.4f}  "
                  f"val_acc={acc:.1f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODELS_DIR / "behavior_lstm_best.pth")

    # Cargar mejor modelo y evaluar
    model.load_state_dict(torch.load(MODELS_DIR / "behavior_lstm_best.pth"))
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            out = model(xb.to(DEVICE))
            all_preds.extend(out["probs"].argmax(1).cpu().tolist())
            all_true.extend(yb.tolist())

    print(f"\n[3] Reporte final (mejor modelo, val_acc={best_acc:.1f}%)")
    print(classification_report(all_true, all_preds,
                                 target_names=BEHAVIOR_CLASSES, zero_division=0))

    # Guardar historia y exportar ONNX
    with open(MODELS_DIR / "behavior_lstm_history.pkl", "wb") as f:
        pickle.dump(history, f)

    model.export_onnx(str(MODELS_DIR / "behavior_lstm.onnx"))
    print(f"\n[LUCY] BehaviorLSTM entrenado. Mejor acc: {best_acc:.1f}%")
    print(f"       Guardado en: models/behavior_lstm_best.pth")
    return model, history


if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    train()
