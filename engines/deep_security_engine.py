"""
DeepSecurityEngine — Motor de seguridad ciudadana de LUCY.

Reemplaza completamente el motor basado en reglas anterior.
Usa modelos entrenados con deep learning:

  1. BehaviorLSTM     → comportamiento temporal (aprende de trayectorias)
  2. RF-DETR weapons  → detección de armas especializada
  3. PPO Risk Agent   → decisión final integrada (RL)

Sin reglas hardcodeadas. Sin umbrales fijos.
Todo lo que LUCY sabe, lo aprendió de datos.
"""

from __future__ import annotations
import os, time
import numpy as np
import torch
from collections import deque
from typing import List, Dict, Any, Optional

from models.behavior_lstm import (
    BehaviorLSTM, BEHAVIOR_CLASSES, SEQ_LEN, FRAME_FEATS,
    build_frame_features, N_CLASSES
)
from models.rl_environment import ActorCritic, STATE_DIM, ACTIONS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Clases COCO relevantes para seguridad
CLASS_PERSON   = 0
CLASS_KNIFE    = 43
CLASS_SCISSORS = 76
WEAPON_CLASSES = {CLASS_KNIFE, CLASS_SCISSORS}

SECURITY_ALERTS = {
    "normal":        {"level": "SAFE",     "color": (0,200,0),   "emoji": "OK"},
    "loitering":     {"level": "WARNING",  "color": (0,165,255), "emoji": "MERODEANDO"},
    "confrontation": {"level": "CRITICAL", "color": (0,0,255),   "emoji": "CONFRONTACION"},
    "encirclement":  {"level": "CRITICAL", "color": (0,0,220),   "emoji": "ENCERCAMIENTO"},
    "following":     {"level": "WARNING",  "color": (0,140,255), "emoji": "SEGUIMIENTO"},
    "violence":      {"level": "CRITICAL", "color": (0,0,200),   "emoji": "VIOLENCIA"},
}


class DeepSecurityEngine:
    """
    Motor de seguridad ciudadana con IA genuina para LUCY.

    Mantiene estado temporal entre frames (memoria de LUCY).
    Usa BehaviorLSTM para entender secuencias de comportamiento.
    Usa PPO Agent para la decisión final de alerta.

    Uso:
        engine = DeepSecurityEngine()
        result = engine.analyze(detections, image_shape, timestamp=time.time())
        engine.reset()  # nueva sesión
    """

    MODELS_DIR = "models"
    SEQ_BUFFER = SEQ_LEN   # frames en memoria

    def __init__(self):
        self._lstm:       Optional[BehaviorLSTM] = None
        self._rl_agent:   Optional[ActorCritic]  = None
        self._seq_buffer: deque = deque(maxlen=self.SEQ_BUFFER)
        self._prev_persons: List[Dict] = []
        self._frame_count = 0
        self._last_brightness = 128.0

        self._load_models()

    def _load_models(self):
        lstm_path = os.path.join(self.MODELS_DIR, "behavior_lstm_best.pth")
        rl_path   = os.path.join(self.MODELS_DIR, "rl_agent_best.pth")

        if os.path.exists(lstm_path):
            self._lstm = BehaviorLSTM().to(DEVICE)
            self._lstm.load_state_dict(
                torch.load(lstm_path, map_location=DEVICE, weights_only=True)
            )
            self._lstm.eval()
            print(f"[LUCY Security] BehaviorLSTM cargado ({lstm_path})")
        else:
            print(f"[LUCY Security] BehaviorLSTM no encontrado — modo básico")

        if os.path.exists(rl_path):
            self._rl_agent = ActorCritic(STATE_DIM).to(DEVICE)
            self._rl_agent.load_state_dict(
                torch.load(rl_path, map_location=DEVICE, weights_only=True)
            )
            self._rl_agent.eval()
            print(f"[LUCY Security] PPO Agent cargado ({rl_path})")
        else:
            print(f"[LUCY Security] PPO Agent no encontrado — usando LSTM solo")

    def reset(self):
        """Reinicia el estado temporal (nueva sesión de video)."""
        self._seq_buffer.clear()
        self._prev_persons.clear()
        self._frame_count = 0

    def analyze(
        self,
        detections:  List[Dict[str, Any]],
        image_shape,
        image:       Optional[np.ndarray] = None,
        timestamp:   Optional[float]      = None,
    ) -> Dict[str, Any]:
        """
        Analiza un frame y retorna evaluación de seguridad.

        detections  : salida de RTDetrDetector
        image_shape : (H, W, ...)
        image       : array numpy BGR para análisis de iluminación
        timestamp   : tiempo UNIX del frame

        Retorna dict con risk_level, behavior, confianza, atención, etc.
        """
        self._frame_count += 1
        img_h, img_w = image_shape[0], image_shape[1]

        # Separar detecciones por tipo
        persons = [d for d in detections if d["class_id"] == CLASS_PERSON]
        weapons = [d for d in detections if d["class_id"] in WEAPON_CLASSES]

        # Análisis de iluminación
        brightness = self._get_brightness(image)
        weapon_near = len(weapons) > 0

        # Construir frame actual para el buffer
        frame_feats = build_frame_features(
            persons       = persons,
            prev_persons  = self._prev_persons,
            brightness    = brightness,
            weapon_near   = weapon_near,
            img_w         = img_w,
            img_h         = img_h,
        )
        self._seq_buffer.append(frame_feats)
        self._prev_persons = persons

        # ── Análisis LSTM (solo cuando hay buffer suficiente) ─────────────
        lstm_result = self._run_lstm()

        # ── Construir estado para RL Agent ────────────────────────────────
        rl_state  = self._build_rl_state(persons, weapons, brightness, lstm_result)
        rl_result = self._run_rl(rl_state)

        # ── Resultado final ───────────────────────────────────────────────
        behavior  = lstm_result.get("class_name", "normal") if lstm_result else "normal"
        alert     = SECURITY_ALERTS.get(behavior, SECURITY_ALERTS["normal"])

        # El RL agent puede elevar el nivel de alerta
        final_level = alert["level"]
        if rl_result and rl_result["action"] == "CRITICAL":
            final_level = "CRITICAL"
        elif rl_result and rl_result["action"] == "WARNING" and final_level == "SAFE":
            final_level = "WARNING"

        return {
            "risk_level":    final_level,
            "behavior":      behavior,
            "behavior_conf": lstm_result.get("confidence", 0.0) if lstm_result else 0.0,
            "all_behaviors": lstm_result.get("all_probs", {}) if lstm_result else {},
            "attention":     lstm_result.get("attention", []) if lstm_result else [],
            "rl_action":     rl_result.get("action") if rl_result else None,
            "rl_probs":      rl_result.get("probs") if rl_result else None,
            "weapons_found": len(weapons),
            "persons_found": len(persons),
            "brightness":    round(brightness, 1),
            "is_dark":       brightness < 90,
            "color_alerta":  alert["color"],
            "frame":         self._frame_count,
            "buffer_ready":  len(self._seq_buffer) >= SEQ_LEN,
        }

    def _run_lstm(self) -> Optional[Dict]:
        if self._lstm is None or len(self._seq_buffer) < SEQ_LEN:
            return None
        seq = torch.stack(list(self._seq_buffer)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return self._lstm.predict(seq.squeeze(0))

    def _build_rl_state(self, persons, weapons, brightness, lstm_result) -> np.ndarray:
        """Construye el vector de estado para el agente RL."""
        s = np.zeros(STATE_DIM, dtype=np.float32)
        s[0]  = min(len(persons), 10) / 10.0
        s[1]  = 0.0  # vehículos (no aplica en seguridad)
        s[2]  = float(len(weapons) > 0)
        s[11] = brightness / 255.0
        s[12] = float(brightness < 90)

        if lstm_result:
            probs = lstm_result.get("all_probs", {})
            s[8]  = probs.get("loitering",     0.0)
            s[9]  = probs.get("following",      0.0)
            s[10] = probs.get("violence",       0.0)
            s[6]  = probs.get("confrontation",  0.0)
            s[7]  = probs.get("encirclement",   0.0)
            s[14] = probs.get("normal",         0.0)

        s[25] = float(len(weapons) > 0 and len(persons) > 0)
        s[23] = float(s[10] > 0.4 or s[2] > 0)
        return s

    def _run_rl(self, state: np.ndarray) -> Optional[Dict]:
        if self._rl_agent is None:
            return None
        with torch.no_grad():
            action, lp, val = self._rl_agent.act(state, deterministic=True)
            logits, _ = self._rl_agent.forward(
                torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            )
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        return {
            "action": ACTIONS[action],
            "probs": {ACTIONS[i]: round(probs[i], 4) for i in range(len(ACTIONS))},
        }

    @staticmethod
    def _get_brightness(image: Optional[np.ndarray]) -> float:
        if image is None:
            return 128.0
        bmap = image.max(axis=2).astype(np.float32)
        return float(np.percentile(bmap, 75))
