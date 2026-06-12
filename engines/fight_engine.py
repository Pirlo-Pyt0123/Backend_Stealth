"""
FightEngine — detección de peleas en tiempo real para LUCY.

Pipeline:
    frame → YOLOv8n-pose → keypoints normalizados
          → buffer 30 frames → FightLSTM (BiLSTM)
          → FIGHT / NO_FIGHT + confianza

El modelo fue entrenado con Real Life Violence Situations (RLVS, 2000 clips).
Accuracy val: 86%. Archivo: models/fight_detector.pth
"""

from __future__ import annotations
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constantes de arquitectura (deben coincidir con el notebook de entrenamiento) ─
N_FRAMES    = 30
N_PEOPLE    = 2
N_KP        = 17
FEATURES    = N_PEOPLE * N_KP * 2    # 68
HIDDEN_SIZE = 256
NUM_LAYERS  = 2
DROPOUT     = 0.4
CLASSES     = ["NoViolencia", "Pelea"]
THRESHOLD   = 0.50    # mínima confianza para declarar FIGHT (sensible, cualquier detección es crítica)


# ── Arquitectura FightLSTM ─────────────────────────────────────────────────────

class FightLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(FEATURES, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.lstm = nn.LSTM(
            input_size    = 128,
            hidden_size   = HIDDEN_SIZE,
            num_layers    = NUM_LAYERS,
            batch_first   = True,
            dropout       = DROPOUT if NUM_LAYERS > 1 else 0.0,
            bidirectional = True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        B, T, _ = x.shape
        proj    = self.input_proj(x.view(B * T, -1)).view(B, T, 128)
        out, _  = self.lstm(proj)
        logits  = self.classifier(out[:, -1, :])
        return {"logits": logits, "probs": F.softmax(logits, dim=-1)}


# ── Normalización de keypoints ─────────────────────────────────────────────────

def _normalize_kp(kp_pixels: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    kp = kp_pixels.astype(np.float32).copy()
    kp[:, 0] /= (img_w + 1e-6)
    kp[:, 1] /= (img_h + 1e-6)

    hip_center      = (kp[11] + kp[12]) / 2.0
    shoulder_center = (kp[5]  + kp[6])  / 2.0
    torso_height    = float(np.linalg.norm(shoulder_center - hip_center)) + 1e-6

    if torso_height < 0.02:
        visible = kp[(kp[:, 0] > 0.01) | (kp[:, 1] > 0.01)]
        if len(visible) >= 2:
            hip_center   = visible.mean(axis=0)
            torso_height = 0.15
        else:
            return kp.flatten()

    return ((kp - hip_center) / torso_height).flatten()


# ── Engine principal ───────────────────────────────────────────────────────────

class FightEngine:
    """Detección de peleas frame a frame con buffer deslizante por sesión."""

    def __init__(
        self,
        model_path: str = "models/fight_detector.pth",
        pose_model: str = "yolov8n-pose.pt",
        device: str | None = None,
    ):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Modelo de clasificación
        self._model = FightLSTM().to(self._device)
        state = torch.load(model_path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

        # YOLOv8n-pose para extracción de keypoints
        from ultralytics import YOLO
        self._pose = YOLO(pose_model)

        # Buffers por sesión: deque de 30 vectores de features (68,)
        self._buffers: dict[str, collections.deque] = {}

        # Último resultado por sesión (evita recalcular antes de tener 30 frames)
        self._last: dict[str, dict] = {}

        print(f"[FightEngine] listo | {self._device} | umbral={THRESHOLD}")

    def analyze(self, image: np.ndarray, session_id: str = "default") -> dict:
        """
        Procesa un frame y devuelve el estado de detección de pelea.
        Devuelve resultado estable hasta que el buffer completa 30 frames.
        """
        if session_id not in self._buffers:
            self._buffers[session_id] = collections.deque(maxlen=N_FRAMES)
            self._last[session_id]    = self._empty_result()

        frame_vec, fight_bboxes = self._extract_keypoints(image)
        self._buffers[session_id].append(frame_vec)

        frames_in   = len(self._buffers[session_id])
        buffer_ready = frames_in >= N_FRAMES

        if buffer_ready:
            seq    = np.stack(list(self._buffers[session_id]))   # (30, 68)
            tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self._device)

            with torch.no_grad():
                out   = self._model(tensor)
                probs = out["probs"][0].cpu().numpy()

            fight_prob = float(probs[1])
            detected   = fight_prob >= THRESHOLD

            result = {
                "fight_detected":   detected,
                "fight_confidence": fight_prob,
                "fight_probs":      {"NoViolencia": float(probs[0]), "Pelea": fight_prob},
                "fight_frames_in":  frames_in,
                "fight_ready":      True,
                "fight_persons":    fight_bboxes,
            }
            self._last[session_id] = result
        else:
            result = dict(self._last[session_id])
            result["fight_frames_in"] = frames_in
            result["fight_ready"]     = False
            result["fight_persons"]   = fight_bboxes   # bboxes actuales aunque el buffer no esté lleno

        return result

    def reset(self, session_id: str = "default") -> None:
        self._buffers.pop(session_id, None)
        self._last.pop(session_id, None)

    def _extract_keypoints(self, image: np.ndarray) -> tuple:
        """Devuelve (vector de features, lista de bboxes [x1,y1,x2,y2])."""
        h, w   = image.shape[:2]
        result = self._pose(image, verbose=False, device=self._device)
        vec    = np.zeros(FEATURES, dtype=np.float32)
        bboxes = []

        if not result or result[0].keypoints is None or len(result[0].boxes) == 0:
            return vec, bboxes

        kp_data = result[0].keypoints.xy.cpu().numpy()    # (n, 17, 2)
        conf    = result[0].boxes.conf.cpu().numpy()       # (n,)
        boxes   = result[0].boxes.xyxy.cpu().numpy()       # (n, 4)
        order   = np.argsort(conf)[::-1]

        for slot, idx in enumerate(order[:N_PEOPLE]):
            kp_norm = _normalize_kp(kp_data[idx], w, h)
            start   = slot * N_KP * 2
            vec[start: start + N_KP * 2] = kp_norm
            x1, y1, x2, y2 = boxes[idx].astype(int).tolist()
            bboxes.append([x1, y1, x2, y2])

        return vec, bboxes

    @staticmethod
    def _empty_result() -> dict:
        return {
            "fight_detected":   False,
            "fight_confidence": 0.0,
            "fight_probs":      {"NoViolencia": 1.0, "Pelea": 0.0},
            "fight_frames_in":  0,
            "fight_ready":      False,
            "fight_persons":    [],
        }
