"""
SuspiciousBehaviorEngine — Motor de detección de comportamiento sospechoso
para StealthVision en tiempo real.

Diseñado para análisis frame-a-frame continuo con memoria temporal.
Integración: Unreal Engine envía frames vía HTTP, el servidor mantiene el estado.

Niveles de alerta:
  SAFE     → Sin amenaza detectada
  WARNING  → Comportamiento sospechoso, vigilar
  CRITICAL → Amenaza inmediata, acción requerida

Arquitectura de dos etapas:
  1. RT-DETR detecta objetos → bboxes + clases
  2. PersonTracker mantiene identidades entre frames (estado temporal)
  3. GradientBoostingClassifier clasifica el escenario a partir del vector de
     features (detecciones + estado del tracker) → scenario_key + probabilidades

Si models/security_clf.pkl no existe, cae automáticamente al análisis por reglas.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, deque
import os
import pickle
import numpy as np
import time


# ---------------------------------------------------------------------------
# Clases COCO relevantes para seguridad
# ---------------------------------------------------------------------------
CLASS_PERSON    = 0
CLASS_BACKPACK  = 24
CLASS_HANDBAG   = 26
CLASS_KNIFE     = 43
CLASS_SCISSORS  = 76

WEAPON_CLASSES   = {CLASS_KNIFE, CLASS_SCISSORS}
OBJECT_CLASSES   = {CLASS_BACKPACK, CLASS_HANDBAG}
SECURITY_CLASSES = {CLASS_PERSON} | WEAPON_CLASSES | OBJECT_CLASSES

# ---------------------------------------------------------------------------
# Biblioteca de escenarios de seguridad
# ---------------------------------------------------------------------------
SECURITY_SCENARIOS: Dict[str, Dict] = {

    "arma_detectada": {
        "risk_level": "CRITICAL",
        "titulo":     "ALERTA: Arma blanca visible",
        "descripcion": (
            "Se detectó un arma blanca (cuchillo o tijeras) cerca de una persona. "
            "Riesgo inmediato de agresión."
        ),
        "accion":     "Activar protocolo de seguridad. Alertar a personal inmediatamente.",
        "consejo":    "OBJETO PELIGROSO DETECTADO — Intervenir ahora",
        "color":      (0, 0, 255),      # Rojo
    },

    "confrontacion": {
        "risk_level": "CRITICAL",
        "titulo":     "ALERTA: Posible confrontación / pelea",
        "descripcion": (
            "Múltiples personas en contacto físico o proximidad extrema. "
            "Patrón consistente con altercado o agresión."
        ),
        "accion":     "Revisar zona. Posible pelea en curso.",
        "consejo":    "CONFRONTACIÓN DETECTADA — Verificar urgente",
        "color":      (0, 0, 220),      # Rojo oscuro
    },

    "encercamiento": {
        "risk_level": "CRITICAL",
        "titulo":     "ALERTA: Persona rodeada",
        "descripcion": (
            "Una persona está siendo rodeada por varias otras desde múltiples ángulos. "
            "Patrón típico de acoso o asalto grupal."
        ),
        "accion":     "Intervenir de inmediato. Posible robo o agresión grupal.",
        "consejo":    "ENCERCAMIENTO DETECTADO — Actuar ahora",
        "color":      (0, 0, 200),      # Rojo
    },

    "merodeo": {
        "risk_level": "WARNING",
        "titulo":     "Advertencia: Persona merodeando",
        "descripcion": (
            "Una persona lleva tiempo estacionada en la misma zona sin moverse. "
            "Comportamiento atípico que puede indicar acecho o reconocimiento del lugar."
        ),
        "accion":     "Vigilar a la persona. Verificar si tiene algún propósito legítimo.",
        "consejo":    "PERSONA INMÓVIL — Monitorear comportamiento",
        "color":      (0, 165, 255),    # Naranja
    },

    "seguimiento_sospechoso": {
        "risk_level": "WARNING",
        "titulo":     "Advertencia: Posible seguimiento",
        "descripcion": (
            "Una persona ha mantenido la misma trayectoria que otra durante "
            "varios segundos. Posible seguimiento o acecho."
        ),
        "accion":     "Monitorear la trayectoria de ambas personas.",
        "consejo":    "TRAYECTORIAS PARALELAS — Vigilar",
        "color":      (0, 140, 255),    # Naranja
    },

    "zona_oscura_persona": {
        "risk_level": "WARNING",
        "titulo":     "Advertencia: Persona en zona oscura",
        "descripcion": (
            "Se detecta actividad humana en un área con poca iluminación. "
            "Las zonas oscuras favorecen comportamientos delictivos."
        ),
        "accion":     "Activar iluminación o verificar la zona.",
        "consejo":    "ZONA OSCURA CON ACTIVIDAD — Revisar",
        "color":      (30, 80, 180),    # Azul oscuro
    },

    "grupo_sospechoso": {
        "risk_level": "WARNING",
        "titulo":     "Advertencia: Grupo numeroso en zona restringida",
        "descripcion": (
            "Se detecta un grupo de personas en concentración inusual. "
            "Puede ser señal de actividad coordinada."
        ),
        "accion":     "Verificar el motivo de la concentración.",
        "consejo":    "CONCENTRACIÓN INUSUAL — Monitorear",
        "color":      (0, 120, 255),    # Naranja suave
    },

    "situacion_normal": {
        "risk_level": "SAFE",
        "titulo":     "Situación normal",
        "descripcion": "No se detectaron comportamientos sospechosos en este frame.",
        "accion":     "Continuar monitoreo.",
        "consejo":    "Sin amenazas detectadas",
        "color":      (0, 200, 0),      # Verde
    },
}


# ---------------------------------------------------------------------------
# Helpers geométricos
# ---------------------------------------------------------------------------

def _center(bbox: List[int]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _iou(a: List[int], b: List[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dist_norm(a: List[int], b: List[int], img_w: int, img_h: int) -> float:
    ca, cb = _center(a), _center(b)
    return ((ca[0] - cb[0]) ** 2 / img_w ** 2 + (ca[1] - cb[1]) ** 2 / img_h ** 2) ** 0.5


def _overlap(a: List[int], b: List[int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _is_surrounded(target: List[int], others: List[List[int]], img_w: int, img_h: int) -> bool:
    """
    Verifica si `target` está rodeado por personas en `others`.
    Estrategia: comprueba que haya personas en al menos 3 cuadrantes distintos
    alrededor del target (norte, sur, este, oeste).
    """
    if len(others) < 3:
        return False

    cx, cy = _center(target)
    quadrants = set()

    for bbox in others:
        ox, oy = _center(bbox)
        dx = ox - cx
        dy = oy - cy
        # Asignamos cuadrante según dirección dominante
        if abs(dx) >= abs(dy):
            quadrants.add("E" if dx > 0 else "W")
        else:
            quadrants.add("S" if dy > 0 else "N")

    return len(quadrants) >= 3


def analyze_lighting(image: np.ndarray) -> Dict[str, Any]:
    brightness_map = image.max(axis=2).astype(np.float32)
    brightness = float(np.percentile(brightness_map, 75))
    if brightness < 50:
        nivel = "oscuro"
        is_dark = True
    elif brightness < 90:
        nivel = "tenue"
        is_dark = True
    else:
        nivel = "normal"
        is_dark = False
    return {"is_dark": is_dark, "brightness": round(brightness, 1), "nivel_luz": nivel}


# ---------------------------------------------------------------------------
# Tracker de personas entre frames
# ---------------------------------------------------------------------------

class PersonTrack:
    """Estado de una persona rastreada entre frames."""

    HISTORY_SECONDS = 8.0    # Cuánto historial guardamos
    LOITER_SECONDS  = 5.0    # Segundos sin moverse → merodeo
    LOITER_THRESH   = 0.06   # Desplazamiento máximo normalizado para "quieto"

    def __init__(self, track_id: int, bbox: List[int], timestamp: float):
        self.track_id   = track_id
        self.bbox       = bbox
        self.last_seen  = timestamp
        self.created_at = timestamp

        # Historial: deque de (timestamp, centro_normalizado)
        self.history: deque = deque()

    def update(self, bbox: List[int], timestamp: float, img_w: int, img_h: int):
        self.bbox      = bbox
        self.last_seen = timestamp
        cx, cy = _center(bbox)
        self.history.append((timestamp, cx / img_w, cy / img_h))
        # Purgar historial antiguo
        cutoff = timestamp - self.HISTORY_SECONDS
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def is_loitering(self, now: float) -> bool:
        """True si la persona lleva LOITER_SECONDS sin moverse significativamente."""
        if not self.history:
            return False
        # Solo considerar si lleva suficiente tiempo siendo rastreado
        if (now - self.created_at) < self.LOITER_SECONDS:
            return False

        recent = [e for e in self.history if e[0] >= now - self.LOITER_SECONDS]
        if len(recent) < 3:
            return False

        xs = [e[1] for e in recent]
        ys = [e[2] for e in recent]
        spread = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        return spread < self.LOITER_THRESH

    @property
    def age_seconds(self) -> float:
        return self.last_seen - self.created_at


class PersonTracker:
    """
    Tracker multi-persona basado en IoU entre frames consecutivos.
    Mantiene identidades estables para análisis temporal.
    """

    MAX_AGE    = 1.5    # Segundos sin aparecer → eliminar track
    IOU_THRESH = 0.25   # IoU mínimo para asociar detección con track existente

    def __init__(self):
        self._tracks: Dict[int, PersonTrack] = {}
        self._next_id = 0

    def update(
        self,
        person_detections: List[Dict[str, Any]],
        timestamp: float,
        img_w: int,
        img_h: int,
    ) -> Dict[int, PersonTrack]:
        """
        Asocia detecciones actuales con tracks existentes.
        Devuelve dict {track_id: PersonTrack} de tracks activos.
        """
        now = timestamp

        # 1. Eliminar tracks demasiado antiguos
        stale = [tid for tid, t in self._tracks.items()
                 if now - t.last_seen > self.MAX_AGE]
        for tid in stale:
            del self._tracks[tid]

        if not person_detections:
            return dict(self._tracks)

        # 2. Asociar detecciones a tracks existentes por IoU greedy
        bboxes = [d["bbox"] for d in person_detections]
        used_tracks = set()
        used_dets   = set()

        if self._tracks:
            track_list = list(self._tracks.values())
            # Matriz de IoU (detecciones × tracks)
            iou_matrix = np.array([
                [_iou(bbox, t.bbox) for t in track_list]
                for bbox in bboxes
            ])

            # Greedy matching: mayor IoU primero
            while True:
                if iou_matrix.size == 0:
                    break
                idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                det_i, trk_j = idx
                if iou_matrix[det_i, trk_j] < self.IOU_THRESH:
                    break
                tid = track_list[trk_j].track_id
                self._tracks[tid].update(bboxes[det_i], now, img_w, img_h)
                used_tracks.add(tid)
                used_dets.add(det_i)
                iou_matrix[det_i, :] = -1
                iou_matrix[:, trk_j] = -1

        # 3. Crear nuevos tracks para detecciones sin asociar
        for i, bbox in enumerate(bboxes):
            if i not in used_dets:
                tid = self._next_id
                self._next_id += 1
                track = PersonTrack(tid, bbox, now)
                track.update(bbox, now, img_w, img_h)
                self._tracks[tid] = track

        return dict(self._tracks)

    def reset(self):
        self._tracks.clear()
        self._next_id = 0


# ---------------------------------------------------------------------------
# Motor principal
# ---------------------------------------------------------------------------

class SuspiciousBehaviorEngine:
    """
    Analiza frames consecutivos de video para detectar comportamientos
    sospechosos en tiempo real.

    El motor mantiene estado temporal entre llamadas:
      - Rastreo de personas (quién es quién entre frames)
      - Detección de merodeo (tiempo sin moverse)
      - Detección de seguimiento (trayectorias paralelas)

    Uso:
        engine = SuspiciousBehaviorEngine()
        result = engine.analyze(detections, image_shape, image=frame, timestamp=time.time())
        engine.reset()   # Para reiniciar entre sesiones
    """

    # Umbrales
    CONFRONTATION_DIST  = 0.10   # Normalizado — personas muy juntas
    ENCIRCLE_DIST       = 0.30   # Distancia máxima para considerar "rodeando"
    WEAPON_PROX_DIST    = 0.25   # Arma cercana a persona
    LARGE_GROUP_COUNT   = 4      # N personas para "grupo sospechoso"

    MODEL_PATH = os.path.join("models", "security_clf.pkl")

    def __init__(self):
        self.tracker = PersonTracker()
        self._frame_count = 0
        self._session_start = time.time()
        self._clf      = None
        self._using_ml = False
        if os.path.exists(self.MODEL_PATH):
            try:
                from engines.feature_extractor import extract_security_features
                self._extract = extract_security_features
                with open(self.MODEL_PATH, "rb") as f:
                    self._clf = pickle.load(f)
                self._using_ml = True
                print(f"SuspiciousBehaviorEngine: modelo ML cargado ({self.MODEL_PATH})")
            except Exception as e:
                print(f"SuspiciousBehaviorEngine: no se pudo cargar ML — usando reglas. ({e})")
        else:
            print("SuspiciousBehaviorEngine: modelo ML no encontrado — usando reglas.")

    def reset(self):
        """Reinicia el estado temporal (nueva sesión de video)."""
        self.tracker.reset()
        self._frame_count = 0
        self._session_start = time.time()

    # ------------------------------------------------------------------
    def analyze(
        self,
        detections: List[Dict[str, Any]],
        image_shape,
        image: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Analiza un frame y devuelve el escenario de seguridad más grave detectado.

        Args:
            detections:   Lista de detecciones del modelo (RT-DETR).
            image_shape:  (height, width, ...) de la imagen.
            image:        Array numpy BGR para análisis de iluminación.
            timestamp:    Tiempo del frame (float UNIX). Si None, usa time.time().
            session_id:   ID de sesión (para futura multi-cámara).

        Returns:
            Dict con: risk_level, scenario_key, titulo, descripcion,
                      accion, consejo, color, tracks_info, stats.
        """
        now = timestamp if timestamp is not None else time.time()
        self._frame_count += 1

        img_h, img_w = image_shape[0], image_shape[1]

        persons  = [d for d in detections if d["class_id"] == CLASS_PERSON]
        weapons  = [d for d in detections if d["class_id"] in WEAPON_CLASSES]
        objects  = [d for d in detections if d["class_id"] in OBJECT_CLASSES]
        lighting = analyze_lighting(image) if image is not None else None

        # El tracker siempre corre — mantiene la identidad de personas entre frames
        active_tracks = self.tracker.update(persons, now, img_w, img_h)

        stats = {
            "personas":       len(persons),
            "armas_visibles": len(weapons),
            "objetos":        len(objects),
            "tracks_activos": len(active_tracks),
            "frame":          self._frame_count,
            "iluminacion":    lighting,
        }

        # ── Path ML ───────────────────────────────────────────────────────
        if self._using_ml:
            feats  = self._extract(detections, image_shape, image, active_tracks, now)
            label  = self._clf.predict(feats.reshape(1, -1))[0]
            probs  = self._clf.predict_proba(feats.reshape(1, -1))[0]
            classes = self._clf.classes_
            scenario_probs = {
                str(cls): round(float(p), 4)
                for cls, p in zip(classes, probs)
            }
            stats["ml_mode"]        = True
            stats["scenario_probs"] = scenario_probs
            result = self._build(label, [], stats, active_tracks)
            result["scenario_probs"] = scenario_probs
            return result

        # ── Fallback: reglas ──────────────────────────────────────────────
        loiterers          = [t for t in active_tracks.values() if t.is_loitering(now)]
        loitering_count    = len(loiterers)
        max_loiter_seconds = max((t.age_seconds for t in loiterers), default=0.0)
        return self._analyze_rules(
            detections, image_shape, image,
            loitering_count=loitering_count,
            max_loiter_seconds=max_loiter_seconds,
            _active_tracks=active_tracks,
            _stats=stats,
        )

    def _analyze_rules(
        self,
        detections: List[Dict[str, Any]],
        image_shape,
        image=None,
        loitering_count: int = 0,
        max_loiter_seconds: float = 0.0,
        _active_tracks: Dict = None,
        _stats: Dict = None,
    ) -> Dict[str, Any]:
        """
        Lógica de reglas pura — usada como fallback en producción y como
        'profesor' para generar etiquetas de entrenamiento ML.

        Parámetros loitering_count y max_loiter_seconds permiten inyectar
        el estado del tracker durante entrenamiento sin instanciar un tracker real.
        """
        img_h, img_w = image_shape[0], image_shape[1]
        active_tracks = _active_tracks or {}

        persons  = [d for d in detections if d["class_id"] == CLASS_PERSON]
        weapons  = [d for d in detections if d["class_id"] in WEAPON_CLASSES]
        objects  = [d for d in detections if d["class_id"] in OBJECT_CLASSES]
        lighting = analyze_lighting(image) if image is not None else None

        stats = _stats or {
            "personas":       len(persons),
            "armas_visibles": len(weapons),
            "objetos":        len(objects),
            "tracks_activos": 0,
            "frame":          0,
            "iluminacion":    lighting,
        }

        scenario_key = "situacion_normal"
        involucrados = []

        # 1. CRITICAL — Arma blanca cerca de persona
        if weapons and persons:
            for weapon in weapons:
                for person in persons:
                    d = _dist_norm(weapon["bbox"], person["bbox"], img_w, img_h)
                    if d < self.WEAPON_PROX_DIST:
                        scenario_key = "arma_detectada"
                        involucrados = [person, weapon]
                        return self._build(scenario_key, involucrados, stats, active_tracks)

        # 2. CRITICAL — Encercamiento (1 persona rodeada por 3+)
        if len(persons) >= 4:
            for i, target in enumerate(persons):
                others = [p for j, p in enumerate(persons) if j != i]
                nearby = [p for p in others
                          if _dist_norm(p["bbox"], target["bbox"], img_w, img_h)
                          < self.ENCIRCLE_DIST]
                if _is_surrounded(target["bbox"], [p["bbox"] for p in nearby], img_w, img_h):
                    scenario_key = "encercamiento"
                    involucrados = [target] + nearby[:4]
                    return self._build(scenario_key, involucrados, stats, active_tracks)

        # 3. CRITICAL — Confrontación (2+ personas con overlap y contacto físico)
        if len(persons) >= 2:
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    if _overlap(persons[i]["bbox"], persons[j]["bbox"]):
                        d = _dist_norm(persons[i]["bbox"], persons[j]["bbox"], img_w, img_h)
                        if d < self.CONFRONTATION_DIST:
                            scenario_key = "confrontacion"
                            involucrados = [persons[i], persons[j]]
                            return self._build(scenario_key, involucrados, stats, active_tracks)

        # 4. WARNING — Merodeo
        if loitering_count > 0:
            scenario_key = "merodeo"
            involucrados = persons[:2]
            stats["merodeo_segundos"] = round(max_loiter_seconds, 1)

        # 5. WARNING — Zona oscura con personas
        if scenario_key == "situacion_normal" and lighting and lighting["is_dark"] and persons:
            scenario_key = "zona_oscura_persona"
            involucrados = persons[:2]

        # 6. WARNING — Grupo numeroso
        if scenario_key == "situacion_normal" and len(persons) >= self.LARGE_GROUP_COUNT:
            scenario_key = "grupo_sospechoso"
            involucrados = persons[:self.LARGE_GROUP_COUNT]

        # 7. Arma visible sin personas cerca — sigue siendo crítico
        if scenario_key == "situacion_normal" and weapons:
            scenario_key = "arma_detectada"
            involucrados = weapons

        return self._build(scenario_key, involucrados, stats, active_tracks)

    # ------------------------------------------------------------------
    def _build(
        self,
        scenario_key: str,
        involucrados: List[Dict],
        stats: Dict,
        active_tracks: Dict,
    ) -> Dict[str, Any]:
        sc = SECURITY_SCENARIOS[scenario_key]

        # Información de tracks para debug / Unreal Engine
        tracks_info = [
            {
                "track_id":       tid,
                "loitering":      t.is_loitering(t.last_seen),
                "age_seconds":    round(t.age_seconds, 1),
                "bbox":           t.bbox,
            }
            for tid, t in active_tracks.items()
        ]

        return {
            "risk_level":    sc["risk_level"],
            "scenario_key":  scenario_key,
            "titulo":        sc["titulo"],
            "descripcion":   sc["descripcion"],
            "accion":        sc["accion"],
            "consejo":       sc["consejo"],
            "color_alerta":  sc["color"],
            "involucrados":  [
                {
                    "clase":      d["class_name"],
                    "confianza":  d["confidence"],
                    "bbox":       d["bbox"],
                }
                for d in involucrados
            ],
            "tracks":  tracks_info,
            "stats":   stats,
        }
