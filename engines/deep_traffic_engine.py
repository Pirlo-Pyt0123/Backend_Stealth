"""
DeepTrafficEngine -- LUCY road risk engine (pedestrian perspective).

TrafficRiskNet = SpatialRiskTransformer + BiGRU.
Trained on JAAD real pedestrian trajectories + synthetic data.

Architecture defined inline -- no external model file dependencies.
Session-aware: each session_id maintains its own GRU frame buffer.

Usage:
    engine = DeepTrafficEngine()
    result = engine.analyze(detections, image_shape, image=frame, session_id="ue5")
"""

from __future__ import annotations
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import List, Dict, Any, Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Architecture constants ─────────────────────────────────────────────────────
N_COCO_CLASSES = 80
D_MODEL        = 64
MAX_OBJECTS    = 20
SEQ_LEN        = 15
GRU_HIDDEN     = 128
GRU_LAYERS     = 2
N_RISK         = 3
RISK_CLASSES   = ["SAFE", "WARNING", "CRITICAL"]

# ── COCO IDs ──────────────────────────────────────────────────────────────────
CLASS_PERSON     = 0
CLASS_BICYCLE    = 1
CLASS_CAR        = 2
CLASS_MOTORCYCLE = 3
CLASS_BUS        = 5
CLASS_TRUCK      = 7
CLASS_TL         = 9
VEHICLE_CLASSES  = {CLASS_CAR, CLASS_MOTORCYCLE, CLASS_BUS, CLASS_TRUCK}
LARGE_VEHICLES   = {CLASS_BUS, CLASS_TRUCK}

RISK_COLORS = {
    "SAFE":     (0, 200, 0),
    "WARNING":  (0, 165, 255),
    "CRITICAL": (0, 0, 255),
}

TL_COLORS_BGR = {
    "RED":     (0,   0,   255),
    "YELLOW":  (0,   215, 255),
    "GREEN":   (0,   200, 0),
    "UNKNOWN": (128, 128, 128),
}

PEDESTRIAN_MESSAGES = {
    "SAFE": {
        "titulo":  "Situacion segura",
        "mensaje": "No se detectaron riesgos para peatones.",
        "leccion": "Siempre mira a ambos lados antes de cruzar.",
    },
    "WARNING": {
        "titulo":  "Precaucion",
        "mensaje": "Situacion que requiere atencion. Observa antes de avanzar.",
        "leccion": "Usa el paso de cebra. Espera a que los vehiculos se detengan.",
    },
    "CRITICAL": {
        "titulo":  "PELIGRO",
        "mensaje": "LUCY detecta riesgo inmediato. Detente y espera.",
        "leccion": "NUNCA cruces sin verificar. Un segundo puede salvar tu vida.",
    },
}


# ── Neural network architecture ────────────────────────────────────────────────
class _ObjectTokenEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_emb = nn.Embedding(N_COCO_CLASSES + 1, 16, padding_idx=N_COCO_CLASSES)
        self.geo_proj  = nn.Linear(5, 32)
        self.out_proj  = nn.Linear(48, D_MODEL)
        self.norm      = nn.LayerNorm(D_MODEL)

    def forward(self, tokens):
        cls_ids  = tokens[..., 0].long().clamp(0, N_COCO_CLASSES)
        geo      = tokens[..., 1:]
        cls_feat = self.class_emb(cls_ids)
        geo_feat = F.relu(self.geo_proj(geo))
        return self.norm(self.out_proj(torch.cat([cls_feat, geo_feat], -1)))


class _SpatialRiskTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = _ObjectTokenEmbedding()
        self.cls_token = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True, norm_first=True)
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),     nn.ReLU(), nn.Linear(64, N_RISK))

    def forward(self, tokens, padding_mask=None, return_features=False):
        B   = tokens.shape[0]
        x   = self.token_emb(tokens)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        if padding_mask is not None:
            cls_mask     = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        enc = self.encoder(x, src_key_padding_mask=padding_mask)
        cls_out = enc[:, 0, :]
        if return_features:
            return cls_out
        logits = self.classifier(cls_out)
        return {"logits": logits, "probs": F.softmax(logits, -1)}


class _GRUTemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=D_MODEL, hidden_size=GRU_HIDDEN,
            num_layers=GRU_LAYERS, batch_first=True,
            dropout=0.3, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(GRU_HIDDEN * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),             nn.ReLU(), nn.Linear(64, N_RISK))

    def forward(self, x):
        out, _ = self.gru(x)
        logits = self.classifier(out[:, -1, :])
        return {"logits": logits, "probs": F.softmax(logits, -1)}


class _TrafficRiskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial  = _SpatialRiskTransformer()
        self.temporal = _GRUTemporalEncoder()

    def forward(self, tokens_seq, masks_seq):
        B, T, N, F = tokens_seq.shape
        feats = self.spatial(
            tokens_seq.view(B * T, N, F),
            masks_seq.view(B * T, N),
            return_features=True,
        )
        return self.temporal(feats.view(B, T, D_MODEL))


# ── Helpers ────────────────────────────────────────────────────────────────────
def _detections_to_tokens(detections, img_w, img_h):
    rows = []
    for det in detections[:MAX_OBJECTS]:
        b  = det["bbox"]
        cx = (b[0] + b[2]) / 2 / img_w
        cy = (b[1] + b[3]) / 2 / img_h
        w  = (b[2] - b[0]) / img_w
        h  = (b[3] - b[1]) / img_h
        rows.append([float(det["class_id"]), cx, cy, w, h, float(det["confidence"])])
    n    = len(rows)
    rows += [[float(N_COCO_CLASSES), 0, 0, 0, 0, 0]] * (MAX_OBJECTS - n)
    return (
        torch.tensor(rows, dtype=torch.float32),
        torch.tensor([False] * n + [True] * (MAX_OBJECTS - n), dtype=torch.bool),
    )


def _detect_tl_state(image: np.ndarray, bbox) -> str:
    """HSV crop analysis: returns 'RED', 'YELLOW', 'GREEN', or 'UNKNOWN'."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(image.shape[1] - 1, x2); y2 = min(image.shape[0] - 1, y2)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 3:
        return "UNKNOWN"
    hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    third = max(1, crop.shape[0] // 3)
    top, mid, bot = hsv[:third], hsv[third:2*third], hsv[2*third:]

    def _red(h):
        return int(cv2.countNonZero(cv2.inRange(h, (0, 100, 100),   (10, 255, 255))) +
                   cv2.countNonZero(cv2.inRange(h, (160, 100, 100), (180, 255, 255))))
    def _yellow(h):
        return int(cv2.countNonZero(cv2.inRange(h, (18, 100, 100), (35, 255, 255))))
    def _green(h):
        return int(cv2.countNonZero(cv2.inRange(h, (40, 80, 80),   (90, 255, 255))))

    scores = {"RED": _red(top), "YELLOW": _yellow(mid), "GREEN": _green(bot)}
    best   = max(scores, key=scores.get)
    return best if scores[best] >= 5 else "UNKNOWN"


# ── Sequence buffer ────────────────────────────────────────────────────────────
class _SequenceBuffer:
    """Sliding window of SEQ_LEN detection frames for the GRU."""

    def __init__(self):
        self.buffer = deque(maxlen=SEQ_LEN)

    @property
    def ready(self) -> bool:
        return len(self.buffer) == SEQ_LEN

    @property
    def size(self) -> int:
        return len(self.buffer)

    def push(self, detections, img_w, img_h):
        tokens, mask = _detections_to_tokens(detections, img_w, img_h)
        self.buffer.append((tokens, mask))

    def predict_full(self, model: _TrafficRiskNet) -> dict:
        """GRU prediction over the full SEQ_LEN window."""
        tokens_seq = torch.stack([b[0] for b in self.buffer]).unsqueeze(0).to(DEVICE)
        masks_seq  = torch.stack([b[1] for b in self.buffer]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(tokens_seq, masks_seq)
        idx = out["probs"].argmax(-1).item()
        return {
            "risk_level": RISK_CLASSES[idx],
            "confidence": float(out["probs"][0, idx]),
            "all_probs":  {RISK_CLASSES[i]: round(float(out["probs"][0, i]), 3) for i in range(N_RISK)},
        }

    def predict_spatial(self, model: _TrafficRiskNet) -> dict:
        """Transformer-only prediction on the last frame (warming-up fallback)."""
        tok, mask = self.buffer[-1]
        with torch.no_grad():
            out = model.spatial(
                tok.unsqueeze(0).to(DEVICE),
                mask.unsqueeze(0).to(DEVICE),
            )
        idx = out["probs"].argmax(-1).item()
        return {
            "risk_level": RISK_CLASSES[idx],
            "confidence": float(out["probs"][0, idx]),
            "all_probs":  {RISK_CLASSES[i]: round(float(out["probs"][0, i]), 3) for i in range(N_RISK)},
        }

    def reset(self):
        self.buffer.clear()


# ── Public engine ──────────────────────────────────────────────────────────────
class DeepTrafficEngine:
    """
    Road risk engine for LUCY — pedestrian perspective.

    Model priority:
        1. models/traffic_risk_jaad.pth  (JAAD + synthetic, Notebook 3)
        2. models/traffic_risk_best.pth  (synthetic only, Notebook 1)
        3. Basic heuristic fallback (no weights found)
    """

    MODELS_DIR       = "models"
    MODEL_CANDIDATES = ["traffic_risk_jaad.pth", "traffic_risk_best.pth"]

    def __init__(self):
        self._model:    Optional[_TrafficRiskNet]        = None
        self._buffers:  Dict[str, _SequenceBuffer]       = {}
        self._model_name = "none"
        self._load_model()

    # ── Setup ──────────────────────────────────────────────────────────────────
    def _load_model(self):
        for fname in self.MODEL_CANDIDATES:
            path = os.path.join(self.MODELS_DIR, fname)
            if os.path.exists(path):
                self._model = _TrafficRiskNet().to(DEVICE)
                self._model.load_state_dict(
                    torch.load(path, map_location=DEVICE, weights_only=True)
                )
                self._model.eval()
                self._model_name = fname
                print(f"[LUCY Traffic] TrafficRiskNet loaded: {fname}  device={DEVICE}")
                return
        print("[LUCY Traffic] No weights found -- heuristic fallback active")

    def _get_buffer(self, session_id: str) -> _SequenceBuffer:
        if session_id not in self._buffers:
            self._buffers[session_id] = _SequenceBuffer()
        return self._buffers[session_id]

    def reset(self, session_id: str = "default"):
        if session_id in self._buffers:
            self._buffers[session_id].reset()

    def session_info(self, session_id: str) -> dict:
        buf = self._get_buffer(session_id)
        return {"frames": buf.size, "ready": buf.ready, "seq_len": SEQ_LEN}

    # ── Main entry point ───────────────────────────────────────────────────────
    def analyze(
        self,
        detections:  List[Dict[str, Any]],
        image_shape,
        image:       Optional[np.ndarray] = None,
        session_id:  str = "default",
    ) -> Dict[str, Any]:
        """
        Evaluate pedestrian road risk for a single frame.

        detections : list of dicts with keys: class_id, bbox, confidence, class_name
        image_shape: (H, W, ...)
        image      : optional BGR numpy array (enables TL HSV detection)
        session_id : identifies the GRU frame buffer (one per UE5 session)
        """
        img_h, img_w = image_shape[0], image_shape[1]
        brightness   = _get_brightness(image)

        # ── Enrich TL detections with HSV state ────────────────────────────
        if image is not None:
            for det in detections:
                if det["class_id"] == CLASS_TL:
                    det["tl_state"] = _detect_tl_state(image, det["bbox"])

        persons  = [d for d in detections if d["class_id"] == CLASS_PERSON]
        bicycles = [d for d in detections if d["class_id"] == CLASS_BICYCLE]
        vehicles = [d for d in detections if d["class_id"] in VEHICLE_CLASSES]
        large_v  = [d for d in detections if d["class_id"] in LARGE_VEHICLES]
        tl_dets  = [d for d in detections if d["class_id"] == CLASS_TL]

        # ── Model inference ────────────────────────────────────────────────
        buf = self._get_buffer(session_id)

        if self._model is not None and detections:
            buf.push(detections, img_w, img_h)
            if buf.ready:
                result = buf.predict_full(self._model)
            else:
                result = buf.predict_spatial(self._model)
        elif self._model is not None:
            # No detections this frame — still push empty frame to keep buffer moving
            buf.push([], img_w, img_h)
            result = {"risk_level": "SAFE", "confidence": 0.95,
                      "all_probs": {"SAFE": 0.95, "WARNING": 0.04, "CRITICAL": 0.01}}
        else:
            # Heuristic fallback (no weights)
            if persons and vehicles:
                result = {"risk_level": "WARNING", "confidence": 0.60,
                          "all_probs": {"SAFE": 0.20, "WARNING": 0.60, "CRITICAL": 0.20}}
            else:
                result = {"risk_level": "SAFE", "confidence": 0.90,
                          "all_probs": {"SAFE": 0.90, "WARNING": 0.08, "CRITICAL": 0.02}}

        risk_level = result["risk_level"]
        confidence = result["confidence"]
        all_probs  = result["all_probs"]

        # ── TL context override ────────────────────────────────────────────
        tl_override = False
        has_red_tl  = any(d.get("tl_state") == "RED" for d in tl_dets)
        if has_red_tl and persons and risk_level == "SAFE":
            risk_level  = "WARNING"
            tl_override = True

        msg = PEDESTRIAN_MESSAGES[risk_level]

        return {
            # Core output (UE5-compatible keys)
            "risk_level":    risk_level,
            "confidence":    round(confidence, 4),
            "all_probs":     all_probs,
            "titulo":        msg["titulo"],
            "mensaje":       msg["mensaje"],
            "leccion":       msg["leccion"],
            "color_alerta":  list(RISK_COLORS[risk_level]),
            # Extended info
            "tl_override":   tl_override,
            "tl_states":     [d.get("tl_state", "UNKNOWN") for d in tl_dets],
            "model_name":    self._model_name,
            "buffer_ready":  buf.ready,
            "frames_in":     buf.size,
            "seq_len":       SEQ_LEN,
            "stats": {
                "personas":   len(persons),
                "bicicletas": len(bicycles),
                "vehiculos":  len(vehicles),
                "grandes":    len(large_v),
                "semaforos":  len(tl_dets),
                "brightness": round(brightness, 1),
                "is_dark":    brightness < 90,
                "total_dets": len(detections),
            },
        }


def _get_brightness(image: Optional[np.ndarray]) -> float:
    if image is None:
        return 128.0
    return float(np.percentile(image.max(axis=2).astype(np.float32), 75))
