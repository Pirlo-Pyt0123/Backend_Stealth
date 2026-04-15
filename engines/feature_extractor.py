"""
feature_extractor.py — Extrae vectores de características numéricos
a partir de las detecciones de RT-DETR para los clasificadores ML de StealthVision.

Cada frame de detección se convierte en un array float32 fijo que el clasificador
usa para predecir el escenario de riesgo y sus probabilidades.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional

# ── IDs de clase COCO ──────────────────────────────────────────────────────
CLASS_PERSON        = 0
CLASS_BICYCLE       = 1
CLASS_CAR           = 2
CLASS_MOTORCYCLE    = 3
CLASS_BUS           = 5
CLASS_TRUCK         = 7
CLASS_TRAFFIC_LIGHT = 9
CLASS_STOP_SIGN     = 11
CLASS_BACKPACK      = 24
CLASS_HANDBAG       = 26
CLASS_KNIFE         = 43
CLASS_SCISSORS      = 76

VEHICLE_CLASSES = {CLASS_CAR, CLASS_MOTORCYCLE, CLASS_BUS, CLASS_TRUCK}
LARGE_VEHICLES  = {CLASS_BUS, CLASS_TRUCK}
WEAPON_CLASSES  = {CLASS_KNIFE, CLASS_SCISSORS}

# ── Nombres de features (para logs e interpretabilidad) ────────────────────
TRAFFIC_FEATURE_NAMES = [
    "n_persons",
    "n_vehicles",
    "n_bicycles",
    "n_large_vehicles",
    "n_traffic_lights",
    "n_stop_signs",
    "min_person_vehicle_dist",       # distancia norm. mínima persona↔vehículo
    "max_person_danger_zone_iou",    # IoU máximo persona vs zona-peligro vehículo expandida
    "person_vehicle_direct_overlap", # 1 si algún bbox persona solapa bbox vehículo
    "min_bicycle_largeveh_dist",     # distancia norm. mínima bicicleta↔vehículo grande
    "brightness_norm",               # brillo p75 normalizado 0-1
    "is_dark",                       # 1 si iluminación insuficiente
    "person_vehicle_ratio",          # n_persons / (n_vehicles + 1)
    "total_detections",
    "both_person_and_vehicle",       # 1 si hay al menos 1 persona Y 1 vehículo
    "bicycle_and_vehicle",           # 1 si hay bicicleta Y vehículo
    "dense_traffic",                 # 1 si ≥2 personas y ≥2 vehículos
    "large_vehicle_present",         # 1 si hay bus o camión
]

SECURITY_FEATURE_NAMES = [
    "n_persons",
    "n_weapons",
    "n_suspicious_objects",          # mochilas, bolsos
    "min_weapon_person_dist",        # distancia norm. mínima arma↔persona
    "weapon_person_direct_overlap",  # 1 si arma solapa persona
    "min_person_person_dist",        # distancia norm. mínima entre personas
    "max_person_person_iou",         # IoU máximo entre dos personas
    "n_overlapping_person_pairs",    # pares de personas con overlap físico cercano
    "encirclement_detected",         # 1 si alguna persona está rodeada ≥3 cuadrantes
    "loitering_count",               # tracks con merodeo activo
    "max_loiter_seconds_norm",       # segundos de merodeo máx, norm. a 10s
    "brightness_norm",
    "is_dark",
    "large_group",                   # 1 si ≥4 personas
    "weapon_with_persons",           # 1 si hay arma Y personas
]


# ── Helpers geométricos ────────────────────────────────────────────────────

def _center(bbox: List[int]):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _dist_norm(a, b, img_w, img_h):
    ca, cb = _center(a), _center(b)
    return ((ca[0] - cb[0]) ** 2 / img_w ** 2
            + (ca[1] - cb[1]) ** 2 / img_h ** 2) ** 0.5


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _bbox_expand(bbox, factor, img_w, img_h):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w  = (x2 - x1) * factor
    h  = (y2 - y1) * factor
    return [
        max(0, int(cx - w / 2)),
        max(0, int(cy - h / 2)),
        min(img_w - 1, int(cx + w / 2)),
        min(img_h - 1, int(cy + h / 2)),
    ]


def _is_surrounded(target, others, img_w, img_h):
    """True si hay personas en al menos 3 cuadrantes distintos alrededor de target."""
    if len(others) < 3:
        return False
    cx, cy = _center(target)
    quadrants = set()
    for bbox in others:
        ox, oy = _center(bbox)
        dx, dy = ox - cx, oy - cy
        if abs(dx) >= abs(dy):
            quadrants.add("E" if dx > 0 else "W")
        else:
            quadrants.add("S" if dy > 0 else "N")
    return len(quadrants) >= 3


def _lighting(image):
    """Devuelve (brightness_p75, is_dark). Usa 128 si no hay imagen."""
    if image is None:
        return 128.0, False
    bmap = image.max(axis=2).astype(np.float32)
    brightness = float(np.percentile(bmap, 75))
    return brightness, brightness < 90


# ── Extractor de tráfico ───────────────────────────────────────────────────

def extract_traffic_features(
    detections: List[Dict[str, Any]],
    image_shape,
    image=None,
) -> np.ndarray:
    """
    Convierte una lista de detecciones en un vector float32 para el
    clasificador de riesgo vial.

    Retorna array de forma (len(TRAFFIC_FEATURE_NAMES),).
    """
    img_h, img_w = image_shape[0], image_shape[1]

    persons    = [d for d in detections if d["class_id"] == CLASS_PERSON]
    bicycles   = [d for d in detections if d["class_id"] == CLASS_BICYCLE]
    vehicles   = [d for d in detections if d["class_id"] in VEHICLE_CLASSES]
    large_vehs = [d for d in detections if d["class_id"] in LARGE_VEHICLES]
    lights     = [d for d in detections if d["class_id"] == CLASS_TRAFFIC_LIGHT]
    stops      = [d for d in detections if d["class_id"] == CLASS_STOP_SIGN]

    brightness, is_dark = _lighting(image)

    # Distancias persona ↔ vehículo
    min_pv_dist       = 1.0
    max_pv_danger_iou = 0.0
    pv_direct_overlap = 0.0
    for p in persons:
        for v in vehicles:
            d = _dist_norm(p["bbox"], v["bbox"], img_w, img_h)
            min_pv_dist = min(min_pv_dist, d)
            danger = _bbox_expand(v["bbox"], 1.6, img_w, img_h)
            max_pv_danger_iou = max(max_pv_danger_iou, _iou(p["bbox"], danger))
            if _overlap(p["bbox"], v["bbox"]):
                pv_direct_overlap = 1.0

    # Distancias bicicleta ↔ vehículo grande
    min_bl_dist = 1.0
    for b in bicycles:
        for lv in large_vehs:
            min_bl_dist = min(min_bl_dist,
                              _dist_norm(b["bbox"], lv["bbox"], img_w, img_h))

    n_p = len(persons)
    n_v = len(vehicles)
    n_b = len(bicycles)
    n_l = len(large_vehs)

    return np.array([
        float(n_p),
        float(n_v),
        float(n_b),
        float(n_l),
        float(len(lights)),
        float(len(stops)),
        min_pv_dist,
        max_pv_danger_iou,
        pv_direct_overlap,
        min_bl_dist,
        brightness / 255.0,
        float(is_dark),
        float(n_p) / (n_v + 1),
        float(len(detections)),
        float(n_p > 0 and n_v > 0),
        float(n_b > 0 and n_v > 0),
        float(n_p >= 2 and n_v >= 2),
        float(n_l > 0),
    ], dtype=np.float32)


# ── Extractor de seguridad ─────────────────────────────────────────────────

def extract_security_features(
    detections: List[Dict[str, Any]],
    image_shape,
    image=None,
    active_tracks=None,
    now: float = None,
) -> np.ndarray:
    """
    Convierte una lista de detecciones + estado del tracker en un vector float32
    para el clasificador de comportamiento sospechoso.

    Retorna array de forma (len(SECURITY_FEATURE_NAMES),).
    """
    img_h, img_w = image_shape[0], image_shape[1]

    persons = [d for d in detections if d["class_id"] == CLASS_PERSON]
    weapons = [d for d in detections if d["class_id"] in WEAPON_CLASSES]
    objects = [d for d in detections
               if d["class_id"] in {CLASS_BACKPACK, CLASS_HANDBAG}]

    brightness, is_dark = _lighting(image)

    # Distancias arma ↔ persona
    min_wp_dist       = 1.0
    wp_direct_overlap = 0.0
    for w in weapons:
        for p in persons:
            d = _dist_norm(w["bbox"], p["bbox"], img_w, img_h)
            min_wp_dist = min(min_wp_dist, d)
            if _overlap(w["bbox"], p["bbox"]):
                wp_direct_overlap = 1.0

    # Distancias persona ↔ persona
    min_pp_dist     = 1.0
    max_pp_iou      = 0.0
    n_overlap_pairs = 0
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            d   = _dist_norm(persons[i]["bbox"], persons[j]["bbox"], img_w, img_h)
            iou = _iou(persons[i]["bbox"], persons[j]["bbox"])
            min_pp_dist = min(min_pp_dist, d)
            max_pp_iou  = max(max_pp_iou, iou)
            if _overlap(persons[i]["bbox"], persons[j]["bbox"]) and d < 0.10:
                n_overlap_pairs += 1

    # Encercamiento
    encirclement = 0.0
    if len(persons) >= 4:
        for i, target in enumerate(persons):
            others = [p for j, p in enumerate(persons) if j != i]
            nearby = [p for p in others
                      if _dist_norm(p["bbox"], target["bbox"], img_w, img_h) < 0.30]
            if _is_surrounded(target["bbox"], [p["bbox"] for p in nearby], img_w, img_h):
                encirclement = 1.0
                break

    # Merodeo — viene del tracker en tiempo real
    loitering_count    = 0
    max_loiter_seconds = 0.0
    if active_tracks and now is not None:
        for t in active_tracks.values():
            if t.is_loitering(now):
                loitering_count += 1
                max_loiter_seconds = max(max_loiter_seconds, t.age_seconds)

    n_p = len(persons)
    n_w = len(weapons)

    return np.array([
        float(n_p),
        float(n_w),
        float(len(objects)),
        min_wp_dist,
        wp_direct_overlap,
        min_pp_dist,
        max_pp_iou,
        float(n_overlap_pairs),
        encirclement,
        float(loitering_count),
        min(max_loiter_seconds / 10.0, 1.0),
        brightness / 255.0,
        float(is_dark),
        float(n_p >= 4),
        float(n_p >= 2 and n_w > 0),
    ], dtype=np.float32)
