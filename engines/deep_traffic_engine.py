"""
DeepTrafficEngine — Motor de riesgo vial de LUCY (perspectiva peatón).

Reemplaza el motor basado en reglas anterior.
Usa el SpatialRiskTransformer: un Transformer que aprendió
qué configuraciones espaciales son peligrosas para un peatón.

Sin distancias hardcodeadas. Sin umbrales geométricos fijos.
La red aprendió sola de miles de escenarios.
"""

from __future__ import annotations
import os
import numpy as np
import torch
from typing import List, Dict, Any, Optional

from models.spatial_risk_net import (
    SpatialRiskTransformer, RISK_CLASSES, detections_to_tokens
)
from models.rl_environment import ActorCritic, STATE_DIM, ACTIONS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# COCO IDs
CLASS_PERSON    = 0
CLASS_BICYCLE   = 1
CLASS_CAR       = 2
CLASS_MOTORCYCLE= 3
CLASS_BUS       = 5
CLASS_TRUCK     = 7
CLASS_TL        = 9
CLASS_STOP      = 11
VEHICLE_CLASSES = {CLASS_CAR, CLASS_MOTORCYCLE, CLASS_BUS, CLASS_TRUCK}
LARGE_VEHICLES  = {CLASS_BUS, CLASS_TRUCK}

RISK_COLORS = {
    "SAFE":     (0, 200, 0),
    "WARNING":  (0, 165, 255),
    "CRITICAL": (0, 0, 255),
}

# Mensajes educativos (perspectiva peatón) — LUCY los explica
PEDESTRIAN_MESSAGES = {
    "SAFE": {
        "titulo":  "Situación segura",
        "mensaje": "No se detectaron riesgos para peatones en esta escena.",
        "leccion": "¡Bien! Pero siempre mira a ambos lados antes de cruzar.",
    },
    "WARNING": {
        "titulo":  "Precaución para peatones",
        "mensaje": "Hay una situación que requiere atención. Reduce velocidad y observa.",
        "leccion": "Usa el paso de cebra. Espera a que los vehículos se detengan completamente.",
    },
    "CRITICAL": {
        "titulo":  "¡PELIGRO para peatón!",
        "mensaje": "LUCY detecta riesgo inmediato. Detente y espera.",
        "leccion": "NUNCA cruces sin verificar. Un segundo de atención puede salvar tu vida.",
    },
}


class DeepTrafficEngine:
    """
    Motor de riesgo vial con IA genuina para LUCY (perspectiva peatón).

    Usa SpatialRiskTransformer para aprender relaciones espaciales
    entre peatones y vehículos sin reglas hardcodeadas.

    Uso:
        engine = DeepTrafficEngine()
        result = engine.analyze(detections, image_shape=(h,w), image=frame)
    """

    MODELS_DIR = "models"

    def __init__(self):
        self._transformer: Optional[SpatialRiskTransformer] = None
        self._rl_agent:    Optional[ActorCritic]            = None
        self._load_models()

    def _load_models(self):
        t_path  = os.path.join(self.MODELS_DIR, "spatial_risk_best.pth")
        rl_path = os.path.join(self.MODELS_DIR, "rl_agent_best.pth")

        if os.path.exists(t_path):
            self._transformer = SpatialRiskTransformer().to(DEVICE)
            self._transformer.load_state_dict(
                torch.load(t_path, map_location=DEVICE, weights_only=True)
            )
            self._transformer.eval()
            print(f"[LUCY Traffic] SpatialRiskTransformer cargado")
        else:
            print(f"[LUCY Traffic] SpatialRiskTransformer no encontrado — modo básico")

        if os.path.exists(rl_path):
            self._rl_agent = ActorCritic(STATE_DIM).to(DEVICE)
            self._rl_agent.load_state_dict(
                torch.load(rl_path, map_location=DEVICE, weights_only=True)
            )
            self._rl_agent.eval()
            print(f"[LUCY Traffic] PPO Agent cargado")

    def analyze(
        self,
        detections:  List[Dict[str, Any]],
        image_shape,
        image:       Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evalúa el riesgo vial para peatones en la escena.

        detections  : lista de dicts (class_id, bbox, confidence, class_name)
        image_shape : (H, W, ...)
        image       : array numpy BGR opcional (para análisis de iluminación)
        """
        img_h, img_w = image_shape[0], image_shape[1]

        persons  = [d for d in detections if d["class_id"] == CLASS_PERSON]
        bicycles = [d for d in detections if d["class_id"] == CLASS_BICYCLE]
        vehicles = [d for d in detections if d["class_id"] in VEHICLE_CLASSES]
        large_v  = [d for d in detections if d["class_id"] in LARGE_VEHICLES]

        brightness = self._get_brightness(image)

        # ── SpatialRiskTransformer ─────────────────────────────────────────
        transformer_result = None
        if self._transformer is not None and detections:
            transformer_result = self._transformer.predict(detections, img_w, img_h)

        # ── Estado RL ─────────────────────────────────────────────────────
        rl_state  = self._build_rl_state(persons, vehicles, large_v, brightness, transformer_result)
        rl_result = self._run_rl(rl_state)

        # ── Nivel de riesgo final ─────────────────────────────────────────
        if transformer_result:
            risk_level = transformer_result["risk_level"]
            confidence = transformer_result["confidence"]
            all_probs  = transformer_result["all_probs"]
            attention  = transformer_result.get("attention", [])
        else:
            # Sin modelo: estimación básica por presencia de objetos
            if persons and vehicles:
                risk_level, confidence = "WARNING", 0.6
            else:
                risk_level, confidence = "SAFE", 0.9
            all_probs, attention = {}, []

        # RL puede elevar el nivel si su política lo indica
        if rl_result and rl_result["action"] != risk_level:
            rl_levels = {"SAFE": 0, "WARNING": 1, "CRITICAL": 2}
            if rl_levels.get(rl_result["action"], 0) > rl_levels.get(risk_level, 0):
                risk_level = rl_result["action"]

        msg = PEDESTRIAN_MESSAGES[risk_level]

        return {
            "risk_level":   risk_level,
            "confidence":   round(confidence, 4),
            "all_probs":    all_probs,
            "attention":    attention,          # qué objetos activaron la alerta
            "titulo":       msg["titulo"],
            "mensaje":      msg["mensaje"],
            "leccion":      msg["leccion"],
            "color_alerta": RISK_COLORS[risk_level],
            "rl_action":    rl_result.get("action") if rl_result else None,
            "stats": {
                "personas":   len(persons),
                "bicicletas": len(bicycles),
                "vehiculos":  len(vehicles),
                "grandes":    len(large_v),
                "brightness": round(brightness, 1),
                "is_dark":    brightness < 90,
                "total_dets": len(detections),
            },
        }

    def _build_rl_state(self, persons, vehicles, large_v, brightness, transformer_result) -> np.ndarray:
        s = np.zeros(STATE_DIM, dtype=np.float32)
        s[0]  = min(len(persons), 10) / 10.0
        s[1]  = min(len(vehicles), 10) / 10.0
        s[11] = brightness / 255.0
        s[12] = float(brightness < 90)
        s[26] = float(len(large_v) > 0)
        s[27] = float(len(vehicles) >= 3)
        s[28] = float(len(persons) > 0 and len(vehicles) > 0)

        if transformer_result:
            probs = transformer_result.get("all_probs", {})
            s[20] = probs.get("SAFE",     0.0)
            s[21] = probs.get("WARNING",  0.0)
            s[22] = probs.get("CRITICAL", 0.0)
            s[23] = probs.get("CRITICAL", 0.0) + probs.get("WARNING", 0.0) * 0.5

        s[24] = float(len(persons) > 0 and brightness < 90)
        return s

    def _run_rl(self, state: np.ndarray) -> Optional[Dict]:
        if self._rl_agent is None:
            return None
        with torch.no_grad():
            action, _, _ = self._rl_agent.act(state, deterministic=True)
            logits, _    = self._rl_agent.forward(
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
