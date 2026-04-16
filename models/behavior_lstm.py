"""
BehaviorLSTM — Cerebro temporal de LUCY.

Aprende patrones de comportamiento humano desde SECUENCIAS de frames.
Sin reglas hardcodeadas. Sin umbrales fijos.
La red descubre sola qué hace a un comportamiento sospechoso.

Datasets de entrenamiento:
  - Stanford Drone Dataset  → trayectorias reales de peatones
  - Fight Detection Dataset → imágenes de violencia/confrontación
  - Secuencias sintéticas   → merodeo, seguimiento, encercamiento enriquecido

Arquitectura:
  Secuencia de T frames
    → Proyección por frame (Linear + LayerNorm)
    → BiLSTM bidireccional (2 capas) — ve pasado Y futuro
    → Atención Temporal — aprende qué frames importan más
    → Clasificador MLP
    → Clase de comportamiento + probabilidades

Clases:
  0  normal          Movimiento normal, sin amenaza
  1  loitering       Merodeo: persona quieta largo tiempo
  2  confrontation   Confrontación: contacto físico o proximidad extrema
  3  encirclement    Encercamiento: persona rodeada desde múltiples ángulos
  4  following       Seguimiento: trayectoria paralela sostenida
  5  violence        Violencia activa: movimientos bruscos, contacto agresivo
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# ── Constantes públicas ───────────────────────────────────────────────────────
BEHAVIOR_CLASSES = ["normal", "loitering", "confrontation",
                    "encirclement", "following", "violence"]
N_CLASSES   = len(BEHAVIOR_CLASSES)   # 6
SEQ_LEN     = 30                      # frames por secuencia
MAX_PERSONS = 5                       # máx personas rastreadas
# Features por frame:
#   1  n_persons norm
#   30 por persona (5×6): cx, cy, w, h, vx, vy
#   4  contexto: brightness, weapon, min_pp_dist, max_pp_iou
FRAME_FEATS = 1 + MAX_PERSONS * 6 + 4   # = 35


# ─────────────────────────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    """
    Atención soft sobre los T pasos temporales.
    El modelo aprende a ponderar frames según su relevancia.
    Permite visualizar QUÉ momentos de la secuencia activaron la alerta.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (B, T, hidden*2)
        scores  = self.v(torch.tanh(self.W(h))).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=1)                    # (B, T)
        context = (h * weights.unsqueeze(-1)).sum(dim=1)      # (B, hidden*2)
        return context, weights


# ─────────────────────────────────────────────────────────────────────────────
class BehaviorLSTM(nn.Module):
    """
    Red LSTM bidireccional con atención temporal para clasificación
    de comportamiento humano desde secuencias de detecciones.

    Parámetros
    ----------
    input_size  : features por frame (default FRAME_FEATS = 35)
    hidden_size : neuronas LSTM por dirección (default 128)
    num_layers  : capas LSTM apiladas (default 2)
    dropout     : dropout entre capas y en clasificador (default 0.35)
    n_classes   : número de comportamientos (default 6)
    """

    def __init__(
        self,
        input_size:  int   = FRAME_FEATS,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.35,
        n_classes:   int   = N_CLASSES,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.n_classes   = n_classes

        # Proyección entrada → espacio latente normalizado
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
        )

        # BiLSTM — bidireccional da contexto del pasado Y el futuro
        self.lstm = nn.LSTM(
            input_size    = 64,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )

        # Atención temporal aprendida
        self.attention = TemporalAttention(hidden_size)

        # Clasificador MLP profundo
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        x : (B, T, input_size)

        Devuelve dict:
          logits       (B, n_classes)
          probs        (B, n_classes)
          attn_weights (B, T)  — solo si return_attention=True
        """
        proj        = self.input_proj(x)                  # (B, T, 64)
        lstm_out, _ = self.lstm(proj)                     # (B, T, hidden*2)
        context, w  = self.attention(lstm_out)            # (B, hidden*2)
        logits      = self.classifier(context)            # (B, n_classes)
        probs       = F.softmax(logits, dim=-1)

        out = {"logits": logits, "probs": probs}
        if return_attention:
            out["attn_weights"] = w
        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict:
        """
        Inferencia lista para producción.
        x : (T, F) o (1, T, F)
        """
        self.eval()
        device = next(self.parameters()).device
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(device)
        out = self.forward(x, return_attention=True)
        idx = out["probs"].argmax(dim=-1).item()
        return {
            "class_id":   idx,
            "class_name": BEHAVIOR_CLASSES[idx],
            "confidence": float(out["probs"][0, idx]),
            "all_probs": {
                BEHAVIOR_CLASSES[i]: round(float(out["probs"][0, i]), 4)
                for i in range(self.n_classes)
            },
            "attention": out["attn_weights"][0].cpu().tolist(),
        }

    def export_onnx(self, path: str, seq_len: int = SEQ_LEN):
        """Exporta a ONNX para integracion con UE5 NNE plugin."""
        try:
            self.eval()
            dummy = torch.zeros(1, seq_len, FRAME_FEATS)
            torch.onnx.export(
                self, (dummy,), path,
                input_names   = ["sequence"],
                output_names  = ["logits", "probs"],
                dynamic_axes  = {
                    "sequence": {0: "batch", 1: "seq_len"},
                    "logits":   {0: "batch"},
                    "probs":    {0: "batch"},
                },
                opset_version = 17,
            )
            print(f"[LUCY] BehaviorLSTM → ONNX: {path}")
        except Exception as e:
            print(f"[LUCY] ONNX export omitido ({e}). El .pth sigue disponible.")


# ── Helper: construir vector de features para un frame ───────────────────────

def build_frame_features(
    persons:       List[Dict],
    prev_persons:  List[Dict],
    brightness:    float = 128.0,
    weapon_near:   bool  = False,
    img_w:         int   = 640,
    img_h:         int   = 640,
) -> torch.Tensor:
    """
    Convierte las detecciones de un frame en el vector de features
    que espera BehaviorLSTM.

    persons / prev_persons : listas de dicts con 'bbox'[x1,y1,x2,y2]
                             y opcionalmente 'track_id'
    Retorna : tensor (FRAME_FEATS,)
    """
    f: List[float] = []
    n = len(persons)

    # 1. n_persons normalizado
    f.append(min(n, MAX_PERSONS) / MAX_PERSONS)

    # Mapa prev por track_id para velocidad
    prev_map = {p.get("track_id", -i-1): p for i, p in enumerate(prev_persons)}

    # 2. Features por persona (padding si hay menos de MAX_PERSONS)
    for i in range(MAX_PERSONS):
        if i < n:
            p  = persons[i]
            b  = p["bbox"]
            cx = (b[0] + b[2]) / 2.0 / img_w
            cy = (b[1] + b[3]) / 2.0 / img_h
            w  = (b[2] - b[0]) / img_w
            h  = (b[3] - b[1]) / img_h
            tid = p.get("track_id", -(i+1))
            vx = vy = 0.0
            if tid in prev_map:
                pb  = prev_map[tid]["bbox"]
                vx  = cx - (pb[0] + pb[2]) / 2.0 / img_w
                vy  = cy - (pb[1] + pb[3]) / 2.0 / img_h
            f.extend([cx, cy, w, h, vx, vy])
        else:
            f.extend([0.0] * 6)

    # 3. Contexto
    f.append(brightness / 255.0)
    f.append(1.0 if weapon_near else 0.0)

    min_d, max_iou = 1.0, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = persons[i]["bbox"], persons[j]["bbox"]
            ci = ((bi[0]+bi[2])/2/img_w, (bi[1]+bi[3])/2/img_h)
            cj = ((bj[0]+bj[2])/2/img_w, (bj[1]+bj[3])/2/img_h)
            d  = ((ci[0]-cj[0])**2 + (ci[1]-cj[1])**2) ** 0.5
            min_d = min(min_d, d)
            ix1 = max(bi[0],bj[0]); iy1 = max(bi[1],bj[1])
            ix2 = min(bi[2],bj[2]); iy2 = min(bi[3],bj[3])
            inter = max(0,ix2-ix1)*max(0,iy2-iy1)
            area  = (bi[2]-bi[0])*(bi[3]-bi[1])+(bj[2]-bj[0])*(bj[3]-bj[1])-inter
            max_iou = max(max_iou, inter/area if area > 0 else 0.0)
    f.extend([min_d, max_iou])

    return torch.tensor(f, dtype=torch.float32)
