"""
LUCY Server — API HTTP para integración con Unreal Engine 5.4

UE5 Blueprint llama a estos endpoints cada frame.
El servidor corre los 3 modelos de IA y devuelve la decisión de LUCY.

Uso:
    cd c:/Users/LENOVO/Documents/Stealth
    python -m server.lucy_server

Endpoints:
    POST /analyze   → pipeline completo (seguridad + tráfico + RL)
    GET  /health    → verificar que el servidor está vivo
    GET  /status    → estado de los modelos cargados
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from collections import deque
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from models.behavior_lstm import (
    BehaviorLSTM, BEHAVIOR_CLASSES, SEQ_LEN, FRAME_FEATS, build_frame_features
)
from models.spatial_risk_net import (
    SpatialRiskTransformer, RISK_CLASSES, detections_to_tokens
)
from models.rl_environment import ActorCritic, ACTIONS, N_ACTIONS, STATE_DIM

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = Path("models")
HOST       = "0.0.0.0"
PORT       = 8765

print(f"[LUCY] Iniciando servidor en {HOST}:{PORT} | Device: {DEVICE}")

# ── Cargar modelos ─────────────────────────────────────────────────────────────
print("[LUCY] Cargando modelos...")

lstm_model = BehaviorLSTM().to(DEVICE)
lstm_model.load_state_dict(
    torch.load(MODELS_DIR / "behavior_lstm_best.pth", map_location=DEVICE, weights_only=True)
)
lstm_model.eval()
print("  ✓ BehaviorLSTM")

spatial_model = SpatialRiskTransformer().to(DEVICE)
spatial_model.load_state_dict(
    torch.load(MODELS_DIR / "spatial_risk_best.pth", map_location=DEVICE, weights_only=True)
)
spatial_model.eval()
print("  ✓ SpatialRiskTransformer")

rl_agent = ActorCritic(state_dim=STATE_DIM, n_actions=N_ACTIONS).to(DEVICE)
rl_agent.load_state_dict(
    torch.load(MODELS_DIR / "rl_agent_best.pth", map_location=DEVICE, weights_only=True)
)
rl_agent.eval()
print("  ✓ PPO RL Agent")
print(f"[LUCY] Todos los modelos listos.\n")

# ── Buffer temporal para BehaviorLSTM ─────────────────────────────────────────
# Acumula 30 frames para alimentar el LSTM
_seq_buffer: deque = deque(maxlen=SEQ_LEN)
_prev_persons: List[dict] = []


# ── Schemas Pydantic (lo que envía UE5) ───────────────────────────────────────
class Detection(BaseModel):
    class_id:   int
    bbox:       List[float]   # [x1, y1, x2, y2] en píxeles (640×640)
    confidence: float
    track_id:   Optional[int] = None

class AnalyzeRequest(BaseModel):
    detections: List[Detection]  # objetos detectados en el frame actual
    brightness: float = 128.0   # luminosidad media del frame (0-255)
    weapon_near: bool = False    # ¿arma detectada por RT-DETR?
    img_w: int = 640
    img_h: int = 640
    reset_sequence: bool = False # True = nueva escena, limpiar buffer LSTM


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LUCY AI Server",
    description="Servidor de IA para el dron LUCY en Unreal Engine 5.4",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.get("/status")
def status():
    return {
        "models": {
            "behavior_lstm":           "loaded",
            "spatial_risk_transformer":"loaded",
            "ppo_agent":               "loaded",
        },
        "buffer_frames": len(_seq_buffer),
        "seq_len_required": SEQ_LEN,
        "device": DEVICE,
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    global _prev_persons, _seq_buffer

    try:
        return _analyze_inner(req)
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "trace": traceback.format_exc()
        })


def _analyze_inner(req: AnalyzeRequest):
    global _prev_persons, _seq_buffer

    if req.reset_sequence:
        _seq_buffer.clear()
        _prev_persons = []

    # ── Convertir detecciones (compatible Pydantic v1 y v2) ──────────────────
    dets_raw = [
        {
            "class_id":   d.class_id,
            "bbox":       list(d.bbox),
            "confidence": d.confidence,
            "track_id":   d.track_id,
        }
        for d in req.detections
    ]

    # Separar personas para el LSTM
    persons = [d for d in dets_raw if d["class_id"] == 0]

    # ── 1. BehaviorLSTM (temporal) ────────────────────────────────────────────
    frame_feat = build_frame_features(
        persons      = persons,
        prev_persons = _prev_persons,
        brightness   = req.brightness,
        weapon_near  = req.weapon_near,
        img_w        = req.img_w,
        img_h        = req.img_h,
    )
    _seq_buffer.append(frame_feat.numpy())
    _prev_persons = persons

    behavior_result = {
        "class_name": "normal",
        "confidence": 1.0,
        "all_probs":  {c: 0.0 for c in BEHAVIOR_CLASSES},
        "attention":  [],
        "ready":      False,  # False hasta tener 30 frames
    }

    if len(_seq_buffer) == SEQ_LEN:
        seq_tensor = torch.tensor(
            np.array(_seq_buffer), dtype=torch.float32
        )
        behavior_result = lstm_model.predict(seq_tensor)
        behavior_result["ready"] = True

    # ── 2. SpatialRiskTransformer (espacial) ──────────────────────────────────
    spatial_result = spatial_model.predict(dets_raw, req.img_w, req.img_h)

    # ── 3. PPO RL Agent (integrador) ─────────────────────────────────────────
    b_probs = np.array([
        behavior_result["all_probs"].get(c, 0.0) for c in BEHAVIOR_CLASSES
    ], dtype=np.float32)
    r_probs = np.array([
        spatial_result["all_probs"].get(c, 0.0) for c in RISK_CLASSES
    ], dtype=np.float32)

    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[:6]  = b_probs
    state[6:9] = r_probs
    state[9]   = float(max(b_probs[2:]))   # max peligro comportamental
    state[10]  = float(r_probs[2])         # P(CRITICAL)
    state[11]  = float(r_probs[1])         # P(WARNING)
    state[12]  = float(b_probs[0])         # P(normal)
    state[13]  = float(r_probs[0])         # P(SAFE)
    state[14]  = req.brightness / 255.0
    state[15]  = 1.0 if req.weapon_near else 0.0

    with torch.no_grad():
        action, _, _ = rl_agent.act(state, deterministic=True)

    # ACTIONS = ["SAFE","WARNING","CRITICAL"] → normalizar a nombres del HUD
    _ACTION_MAP = {"SAFE": "monitor", "WARNING": "warning", "CRITICAL": "critical"}
    rl_action = _ACTION_MAP.get(ACTIONS[action], "monitor")

    # ── Respuesta para UE5 ────────────────────────────────────────────────────
    return {
        # Decisión principal
        "lucy_action":    rl_action,          # "monitor" | "warning" | "critical"
        "alert_level":    action,             # 0 | 1 | 2  (para Blueprint switch)

        # Módulo comportamiento
        "behavior":       behavior_result.get("class_name", "normal"),
        "behavior_conf":  round(behavior_result.get("confidence", 0.0), 3),
        "behavior_ready": behavior_result.get("ready", False),

        # Módulo riesgo espacial
        "risk_level":     spatial_result["risk_level"],
        "risk_conf":      round(spatial_result["confidence"], 3),

        # Para HUD de LUCY en UE5
        "hud": _build_hud(rl_action, behavior_result, spatial_result, req.weapon_near),

        # Atención (para debug en UE5)
        "attention_objects": spatial_result.get("attention", [])[:5],
    }


def _build_hud(action: str, beh: dict, risk: dict, weapon: bool) -> dict:
    """Genera el texto y color que mostrará LUCY en el HUD de UE5."""
    COLORS = {
        "monitor":  {"r": 0,   "g": 200, "b": 0,   "a": 255},
        "warning":  {"r": 255, "g": 165, "b": 0,   "a": 255},
        "critical": {"r": 255, "g": 0,   "b": 0,   "a": 255},
    }
    MESSAGES = {
        "monitor": {
            "title":   "LUCY — Todo en orden",
            "message": "No se detectan amenazas activas.",
        },
        "warning": {
            "title":   "LUCY — Precaución",
            "message": _warning_message(beh.get("class_name","normal"),
                                        risk["risk_level"], weapon),
        },
        "critical": {
            "title":   "LUCY — ¡ALERTA CRÍTICA!",
            "message": _critical_message(beh.get("class_name","normal"),
                                         risk["risk_level"], weapon),
        },
    }
    return {
        "color":   COLORS[action],
        "title":   MESSAGES[action]["title"],
        "message": MESSAGES[action]["message"],
    }


def _warning_message(behavior: str, risk: str, weapon: bool) -> str:
    if weapon:
        return "Objeto peligroso detectado en la zona."
    if behavior == "following":
        return "Posible seguimiento detectado. Mantente alerta."
    if behavior == "loitering":
        return "Persona merodeando el área."
    if risk == "WARNING":
        return "Vehículos cercanos. Precaución al cruzar."
    return "Situación de precaución detectada."


def _critical_message(behavior: str, risk: str, weapon: bool) -> str:
    if weapon:
        return "¡ARMA DETECTADA! Alejarse de inmediato."
    if behavior == "violence":
        return "¡Violencia activa detectada!"
    if behavior in ["confrontation", "encirclement"]:
        return "¡Confrontación peligrosa detectada!"
    if risk == "CRITICAL":
        return "¡Peligro inminente! Detente."
    return "¡Peligro crítico detectado!"


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
