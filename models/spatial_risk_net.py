"""
SpatialRiskTransformer — Cerebro espacial de LUCY.

Aprende relaciones entre objetos detectados para evaluar riesgo vial
desde la perspectiva del PEATÓN.

Sin distancias hardcodeadas. Sin umbrales fijos.
El Transformer aprende por sí mismo qué configuraciones espaciales
son peligrosas para un peatón.

Los attention heads aprenden a "mirar" las relaciones que importan:
  - Head 1 → distancia peatón ↔ vehículo
  - Head 2 → posición relativa al paso de cebra
  - Head 3 → tamaño/velocidad inferida del vehículo
  (esto emerge durante entrenamiento, no está programado)

Escenarios que detecta (perspectiva peatón):
  SAFE     → Sin riesgo aparente para el peatón
  WARNING  → Peatón en situación de precaución
             (cerca de tráfico, sin cruce, ciclista expuesto, etc.)
  CRITICAL → Peligro inminente para el peatón
             (atropello inminente, zona oscura + tráfico, etc.)

Arquitectura:
  Detecciones → Token Embedding por objeto
    → [CLS] token prepend
    → TransformerEncoder (3 capas, 4 cabezas de atención)
    → Extraer [CLS] → MLP clasificador
    → (SAFE / WARNING / CRITICAL)  + attention weights
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# ── Constantes públicas ───────────────────────────────────────────────────────
RISK_CLASSES  = ["SAFE", "WARNING", "CRITICAL"]
N_RISK        = len(RISK_CLASSES)     # 3
MAX_OBJECTS   = 20                    # máx objetos por escena
N_COCO_CLASSES = 80                   # clases COCO que conoce RT-DETR
D_MODEL       = 64                    # dimensión interna del Transformer


# ── COCO class ids relevantes ─────────────────────────────────────────────────
PEDESTRIAN_CLASSES = {0}              # person
VEHICLE_CLASSES    = {1,2,3,5,7}     # bicycle,car,motorcycle,bus,truck
SIGNAL_CLASSES     = {9,11}          # traffic_light, stop_sign
RELEVANT_CLASSES   = PEDESTRIAN_CLASSES | VEHICLE_CLASSES | SIGNAL_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
class ObjectTokenEmbedding(nn.Module):
    """
    Transforma una detección en un token de dimensión D_MODEL.

    Cada objeto = [class_id, cx, cy, w, h, conf]
    El class_id pasa por un Embedding aprendido (no one-hot rígido).
    Las coords geométricas pasan por una proyección lineal.
    Se combinan y proyectan al espacio D_MODEL.
    """

    def __init__(self, n_classes: int = N_COCO_CLASSES, d_model: int = D_MODEL):
        super().__init__()
        self.class_emb  = nn.Embedding(n_classes + 1, 16, padding_idx=n_classes)
        self.geo_proj   = nn.Linear(5, 32)           # cx,cy,w,h,conf → 32
        self.out_proj   = nn.Linear(16 + 32, d_model)
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens : (B, N, 6)  — [class_id(int), cx, cy, w, h, conf]
        returns: (B, N, D_MODEL)
        """
        cls_ids = tokens[..., 0].long().clamp(0, N_COCO_CLASSES)
        geo     = tokens[..., 1:]                      # (B, N, 5)

        cls_feat = self.class_emb(cls_ids)             # (B, N, 16)
        geo_feat = F.relu(self.geo_proj(geo))          # (B, N, 32)
        combined = torch.cat([cls_feat, geo_feat], -1) # (B, N, 48)
        return self.norm(self.out_proj(combined))       # (B, N, D_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
class SpatialRiskTransformer(nn.Module):
    """
    Transformer encoder para evaluación de riesgo vial desde perspectiva peatonal.

    Parámetros
    ----------
    d_model     : dimensión del Transformer (default 64)
    nhead       : cabezas de atención (default 4)
    num_layers  : capas del encoder (default 3)
    dim_ff      : dimensión feed-forward interna (default 256)
    dropout     : dropout (default 0.1)
    max_objects : máximo de objetos por escena (default MAX_OBJECTS=20)
    n_classes   : clases de riesgo (default 3: SAFE/WARNING/CRITICAL)
    """

    def __init__(
        self,
        d_model:     int   = D_MODEL,
        nhead:       int   = 4,
        num_layers:  int   = 3,
        dim_ff:      int   = 256,
        dropout:     float = 0.1,
        max_objects: int   = MAX_OBJECTS,
        n_classes:   int   = N_RISK,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_objects = max_objects
        self.n_classes   = n_classes

        # Embedding de objetos
        self.token_emb = ObjectTokenEmbedding(d_model=d_model)

        # Token [CLS] aprendido — résumen global de la escena
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Encoder Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_ff,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,   # Pre-LN: más estable en entrenamiento
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Clasificador desde [CLS]
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

        # Proyección para exportar attention maps (UE5 visualization)
        self.attn_proj = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        tokens       : (B, N, 6)  — [class_id, cx, cy, w, h, conf]
                       N puede ser < max_objects si se usa padding
        padding_mask : (B, N+1) bool — True = posición a ignorar (padding)
        return_attention : retorna pesos de atención del [CLS] sobre objetos

        Devuelve dict:
          logits (B, n_classes)
          probs  (B, n_classes)
          attn   (B, N)  — solo si return_attention=True
        """
        B = tokens.shape[0]

        # 1. Embedding de cada objeto detectado
        x = self.token_emb(tokens)                     # (B, N, D)

        # 2. Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)               # (B, N+1, D)

        # 3. Ajustar mask para incluir [CLS] (nunca se ignora)
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # 4. Transformer Encoder — aquí ocurre el aprendizaje de relaciones
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        # 5. Extraer [CLS] como representación de la escena completa
        cls_out = encoded[:, 0, :]                     # (B, D)

        # 6. Clasificar
        logits = self.classifier(cls_out)
        probs  = F.softmax(logits, dim=-1)

        out = {"logits": logits, "probs": probs}

        if return_attention:
            # Atención del [CLS] sobre cada objeto (simplificada via proyección)
            obj_encoded = encoded[:, 1:, :]             # (B, N, D)
            attn_scores = self.attn_proj(obj_encoded).squeeze(-1)  # (B, N)
            if padding_mask is not None:
                obj_mask = padding_mask[:, 1:]
                attn_scores = attn_scores.masked_fill(obj_mask, float("-inf"))
            out["attn"] = F.softmax(attn_scores, dim=-1)

        return out

    @torch.no_grad()
    def predict(self, detections: List[Dict], img_w: int = 640, img_h: int = 640) -> Dict:
        """
        Predicción desde lista de detecciones RT-DETR.

        detections : lista de dicts con class_id, bbox, confidence
        """
        self.eval()
        device = next(self.parameters()).device
        tokens, mask = detections_to_tokens(detections, img_w, img_h, self.max_objects)
        tokens = tokens.unsqueeze(0).to(device)
        mask   = mask.unsqueeze(0).to(device)
        out    = self.forward(tokens, mask, return_attention=True)
        idx    = out["probs"].argmax(dim=-1).item()
        return {
            "risk_level":  RISK_CLASSES[idx],
            "class_id":    idx,
            "confidence":  float(out["probs"][0, idx]),
            "all_probs": {
                RISK_CLASSES[i]: round(float(out["probs"][0, i]), 4)
                for i in range(self.n_classes)
            },
            "attention": out["attn"][0].cpu().tolist(),
        }

    def export_onnx(self, path: str):
        """Exporta a ONNX para UE5 NNE."""
        try:
            self.eval()
            dummy_tokens = torch.zeros(1, MAX_OBJECTS, 6)
            dummy_mask   = torch.zeros(1, MAX_OBJECTS, dtype=torch.bool)
            torch.onnx.export(
                self,
                (dummy_tokens, dummy_mask),
                path,
                input_names  = ["tokens", "padding_mask"],
                output_names = ["logits", "probs"],
                dynamic_axes = {
                    "tokens":       {0: "batch"},
                    "padding_mask": {0: "batch"},
                    "logits":       {0: "batch"},
                    "probs":        {0: "batch"},
                },
                opset_version = 17,
            )
            print(f"[LUCY] SpatialRiskTransformer → ONNX: {path}")
        except Exception as e:
            print(f"[LUCY] ONNX export omitido ({e}). El .pth sigue disponible.")


# ── Utilidades ────────────────────────────────────────────────────────────────

def detections_to_tokens(
    detections: List[Dict],
    img_w: int,
    img_h: int,
    max_objects: int = MAX_OBJECTS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convierte lista de detecciones en tensor de tokens + máscara de padding.

    detections : lista de dicts con class_id, bbox[x1,y1,x2,y2], confidence
    Retorna:
      tokens  (max_objects, 6)   — [class_id, cx, cy, w, h, conf]
      mask    (max_objects,) bool — True = posición de padding
    """
    rows   = []
    for det in detections[:max_objects]:
        b   = det["bbox"]
        cx  = (b[0] + b[2]) / 2.0 / img_w
        cy  = (b[1] + b[3]) / 2.0 / img_h
        w   = (b[2] - b[0]) / img_w
        h   = (b[3] - b[1]) / img_h
        rows.append([float(det["class_id"]), cx, cy, w, h, float(det["confidence"])])

    n    = len(rows)
    pad  = max_objects - n
    rows += [[float(N_COCO_CLASSES), 0, 0, 0, 0, 0]] * pad   # token de padding

    tokens = torch.tensor(rows, dtype=torch.float32)
    mask   = torch.tensor([False]*n + [True]*pad, dtype=torch.bool)
    return tokens, mask
