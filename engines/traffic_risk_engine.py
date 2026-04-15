"""
TrafficRiskEngine — Motor de análisis de riesgo vial para StealthVision
Orientado a educación de niños y jóvenes menores de 18 años.

Niveles de riesgo:
  SAFE     → Situación sin riesgo aparente
  WARNING  → Situación que requiere precaución
  CRITICAL → Peligro inmediato, acción requerida

Arquitectura de dos etapas:
  1. RT-DETR (Transformer) detecta objetos en la imagen → bboxes + clases
  2. GradientBoostingClassifier clasifica el escenario a partir del vector de
     features extraído de esas detecciones → scenario_key + probabilidades

Si el modelo ML no está disponible (models/traffic_clf.pkl ausente), el motor
cae al análisis por reglas como fallback automático.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import pickle
import numpy as np


# ---------------------------------------------------------------------------
# Identificadores COCO que usamos
# ---------------------------------------------------------------------------
CLASS_PERSON       = 0
CLASS_BICYCLE      = 1
CLASS_CAR          = 2
CLASS_MOTORCYCLE   = 3
CLASS_BUS          = 5
CLASS_TRUCK        = 7
CLASS_TRAFFIC_LIGHT = 9
CLASS_STOP_SIGN    = 11

VEHICLE_CLASSES = {CLASS_CAR, CLASS_MOTORCYCLE, CLASS_BUS, CLASS_TRUCK}
LARGE_VEHICLES  = {CLASS_BUS, CLASS_TRUCK}

# ---------------------------------------------------------------------------
# Biblioteca de escenarios educativos
# ---------------------------------------------------------------------------
SCENARIOS: Dict[str, Dict] = {

    "atropello_inminente": {
        "risk_level":   "CRITICAL",
        "titulo":       "¡PELIGRO! Riesgo de atropello",
        "explicacion":  (
            "Hay una persona muy cerca o en contacto directo con un vehículo. "
            "Esto puede causar un accidente grave en segundos."
        ),
        "leccion": (
            "Nunca cruces entre vehículos estacionados o en movimiento. "
            "Usa siempre la senda peatonal y espera que los autos se detengan completamente."
        ),
        "consejo_rapido": "PARA. Mira a ambos lados. Cruza solo cuando sea seguro.",
        "color_alerta":   (0, 0, 255),   # Rojo BGR
    },

    "persona_cerca_vehiculo": {
        "risk_level":   "WARNING",
        "titulo":       "Precaución: Persona cerca del tráfico",
        "explicacion":  (
            "Una persona está demasiado cerca de vehículos en movimiento. "
            "Los conductores pueden no tener tiempo de frenar."
        ),
        "leccion": (
            "Mantén siempre distancia segura de los vehículos. "
            "Camina siempre por la vereda, nunca por la calzada."
        ),
        "consejo_rapido": "Retrocede a la vereda. Deja pasar los autos.",
        "color_alerta":   (0, 165, 255),  # Naranja BGR
    },

    "ciclista_entre_vehiculos": {
        "risk_level":   "WARNING",
        "titulo":       "Precaución: Ciclista en zona de riesgo",
        "explicacion":  (
            "Un ciclista está circulando entre vehículos grandes. "
            "Los conductores de camiones y buses tienen puntos ciegos y pueden no verlo."
        ),
        "leccion": (
            "Si andas en bicicleta, usa el carril exclusivo o ciclovía. "
            "Nunca pases por el costado de camiones o buses."
        ),
        "consejo_rapido": "Usa la ciclovía. Sé visible, usa colores brillantes.",
        "color_alerta":   (0, 165, 255),
    },

    "ciclista_sin_proteccion": {
        "risk_level":   "WARNING",
        "titulo":       "Precaución: Ciclista en la calzada",
        "explicacion":  (
            "Se detectó un ciclista circulando en zona de tráfico vehicular."
        ),
        "leccion": (
            "Siempre usa casco al andar en bicicleta. "
            "Circula por el carril de bicicletas cuando esté disponible."
        ),
        "consejo_rapido": "Casco siempre. Señaliza tus giros con la mano.",
        "color_alerta":   (0, 165, 255),
    },

    "peaton_en_calzada": {
        "risk_level":   "WARNING",
        "titulo":       "Precaución: Peatón en la calzada",
        "explicacion":  (
            "Hay personas caminando por la calzada donde circulan los vehículos."
        ),
        "leccion": (
            "La calzada es exclusiva para vehículos. "
            "Los peatones siempre deben caminar por la vereda."
        ),
        "consejo_rapido": "Sube a la vereda inmediatamente.",
        "color_alerta":   (0, 165, 255),
    },

    "trafico_denso_peaton": {
        "risk_level":   "WARNING",
        "titulo":       "Mucho tráfico: espera antes de cruzar",
        "explicacion":  (
            "Hay varios vehículos y peatones en la misma zona. "
            "Con mucho tráfico es más difícil cruzar con seguridad."
        ),
        "leccion": (
            "Cuando hay mucho tráfico, espera que la calle esté despejada o usa el semáforo. "
            "Nunca corras para cruzar."
        ),
        "consejo_rapido": "Espera. Nunca cruces corriendo.",
        "color_alerta":   (0, 165, 255),
    },

    "zona_oscura": {
        "risk_level":   "WARNING",
        "titulo":       "Precaución: Zona con poca iluminación",
        "explicacion":  (
            "Esta zona está oscura o tiene muy poca luz. "
            "En lugares con poca iluminación es mucho más difícil ver y ser visto por los conductores."
        ),
        "leccion": (
            "Evita caminar solo por callejones o zonas oscuras. "
            "Si debes hacerlo, usa ropa o accesorios reflectantes y lleva una linterna. "
            "Los conductores pueden no verte a tiempo para frenar."
        ),
        "consejo_rapido": "Zona oscura: usa ropa reflectante y mantente alerta.",
        "color_alerta":   (0, 100, 200),  # Naranja oscuro BGR
    },

    "zona_oscura_con_trafico": {
        "risk_level":   "CRITICAL",
        "titulo":       "¡PELIGRO! Zona oscura con tráfico",
        "explicacion":  (
            "Hay poca iluminación Y vehículos en movimiento. "
            "Los conductores tienen muy poca visibilidad y pueden no ver a los peatones."
        ),
        "leccion": (
            "En zonas oscuras con tráfico el riesgo de accidente es muy alto. "
            "Nunca cruces una calle oscura sin asegurarte de que los autos te hayan visto. "
            "Usa siempre ropa clara o reflectante de noche."
        ),
        "consejo_rapido": "PELIGRO: Espera luz o señal antes de cruzar.",
        "color_alerta":   (0, 0, 200),    # Rojo oscuro BGR
    },

    "zona_segura": {
        "risk_level":   "SAFE",
        "titulo":       "Zona segura",
        "explicacion":  (
            "No se detectaron situaciones de riesgo inmediato en esta imagen."
        ),
        "leccion": (
            "¡Bien! Pero recuerda: siempre debes mirar a ambos lados antes de cruzar, "
            "aunque el camino parezca libre."
        ),
        "consejo_rapido": "Sigue las normas de tránsito siempre.",
        "color_alerta":   (0, 200, 0),    # Verde BGR
    },
}


# ---------------------------------------------------------------------------
# Helpers geométricos
# ---------------------------------------------------------------------------

def _iou(a: List[int], b: List[int]) -> float:
    """Intersection over Union entre dos bounding boxes [x1,y1,x2,y2]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_dist_norm(a: List[int], b: List[int], img_w: int, img_h: int) -> float:
    """Distancia euclidiana normalizada entre centros de dos bboxes (0-1)."""
    ca = ((a[0] + a[2]) / 2 / img_w, (a[1] + a[3]) / 2 / img_h)
    cb = ((b[0] + b[2]) / 2 / img_w, (b[1] + b[3]) / 2 / img_h)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


def _bbox_expand(bbox: List[int], factor: float, img_w: int, img_h: int) -> List[int]:
    """Expande un bbox por un factor para zona de proximidad."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * factor, (y2 - y1) * factor
    return [
        max(0, int(cx - w / 2)),
        max(0, int(cy - h / 2)),
        min(img_w - 1, int(cx + w / 2)),
        min(img_h - 1, int(cy + h / 2)),
    ]


def _bboxes_overlap(a: List[int], b: List[int]) -> bool:
    """True si los bboxes se solapan."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def analyze_lighting(image: np.ndarray) -> Dict[str, Any]:
    """
    Analiza el nivel de iluminación de la imagen.

    Devuelve:
      - is_dark:      True si la iluminación es insuficiente
      - brightness:   valor medio de luminosidad (0-255)
      - nivel_luz:    "oscuro" | "tenue" | "normal"
    """
    # V = max(R, G, B) por píxel → brillo perceptual sin necesitar cv2
    brightness_map = image.max(axis=2).astype(np.float32)

    # Percentil 75: las zonas brillantes (cielo, faroles) no enmascaran la oscuridad general
    brightness = float(np.percentile(brightness_map, 75))

    if brightness < 50:
        nivel_luz = "oscuro"
        is_dark   = True
    elif brightness < 90:
        nivel_luz = "tenue"
        is_dark   = True
    else:
        nivel_luz = "normal"
        is_dark   = False

    return {
        "is_dark":    is_dark,
        "brightness": round(brightness, 1),
        "nivel_luz":  nivel_luz,
    }


# ---------------------------------------------------------------------------
# Motor principal
# ---------------------------------------------------------------------------

class TrafficRiskEngine:
    """
    Analiza una lista de detecciones y devuelve una evaluación de riesgo vial
    con mensaje educativo para niños y jóvenes.

    Uso:
        engine = TrafficRiskEngine()
        result = engine.analyze(detections, image_shape=(h, w))
    """

    # Umbral de distancia normalizada para considerar "cerca"
    PROXIMITY_THRESHOLD = 0.18
    DANGER_ZONE_FACTOR  = 1.6
    MODEL_PATH = os.path.join("models", "traffic_clf.pkl")

    def __init__(self):
        self._clf   = None
        self._using_ml = False
        if os.path.exists(self.MODEL_PATH):
            try:
                from engines.feature_extractor import extract_traffic_features
                self._extract = extract_traffic_features
                with open(self.MODEL_PATH, "rb") as f:
                    self._clf = pickle.load(f)
                self._using_ml = True
                print(f"TrafficRiskEngine: modelo ML cargado ({self.MODEL_PATH})")
            except Exception as e:
                print(f"TrafficRiskEngine: no se pudo cargar ML — usando reglas. ({e})")
        else:
            print("TrafficRiskEngine: modelo ML no encontrado — usando reglas.")

    def analyze(
        self,
        detections: List[Dict[str, Any]],
        image_shape,          # (height, width) o (height, width, channels)
        image: Optional[np.ndarray] = None,   # imagen BGR para análisis de iluminación
    ) -> Dict[str, Any]:
        """
        Devuelve un diccionario con:
          - risk_level:      SAFE / WARNING / CRITICAL
          - scenario_key:    clave del escenario detectado
          - titulo:          título del peligro (en español)
          - explicacion:     descripción del peligro
          - leccion:         qué debe hacer el usuario
          - consejo_rapido:  mensaje corto
          - color_alerta:    color BGR para visualización
          - detecciones_clave: lista de objetos involucrados
          - stats:           resumen de objetos detectados
        """
        # ── Path ML (clasificador entrenado) ──────────────────────────────
        if self._using_ml:
            feats  = self._extract(detections, image_shape, image)
            label  = self._clf.predict(feats.reshape(1, -1))[0]
            probs  = self._clf.predict_proba(feats.reshape(1, -1))[0]
            classes = self._clf.classes_
            scenario_probs = {
                str(cls): round(float(p), 4)
                for cls, p in zip(classes, probs)
            }
            # Análisis de iluminación para stats (igual que en reglas)
            lighting = analyze_lighting(image) if image is not None else None
            persons  = [d for d in detections if d["class_id"] == CLASS_PERSON]
            vehicles = [d for d in detections if d["class_id"] in VEHICLE_CLASSES]
            bicycles = [d for d in detections if d["class_id"] == CLASS_BICYCLE]
            stats = {
                "personas":      len(persons),
                "bicicletas":    len(bicycles),
                "vehiculos":     len(vehicles),
                "objetos_total": len(detections),
                "iluminacion":   lighting,
                "ml_mode":       True,
                "scenario_probs": scenario_probs,
            }
            result = self._build_result(label, [], stats)
            result["scenario_probs"] = scenario_probs
            return result

        # ── Fallback: análisis por reglas ─────────────────────────────────
        return self._analyze_rules(detections, image_shape, image)

    def _analyze_rules(
        self,
        detections: List[Dict[str, Any]],
        image_shape,
        image: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Lógica de reglas original — usada como fallback y para generar labels de entrenamiento."""
        img_h, img_w = image_shape[0], image_shape[1]

        # Separar por categoría
        persons    = [d for d in detections if d["class_id"] == CLASS_PERSON]
        bicycles   = [d for d in detections if d["class_id"] == CLASS_BICYCLE]
        vehicles   = [d for d in detections if d["class_id"] in VEHICLE_CLASSES]
        large_vehs = [d for d in detections if d["class_id"] in LARGE_VEHICLES]

        # Análisis de iluminación
        lighting = analyze_lighting(image) if image is not None else None

        stats = {
            "personas":    len(persons),
            "bicicletas":  len(bicycles),
            "vehiculos":   len(vehicles),
            "objetos_total": len(detections),
            "iluminacion": lighting,
        }

        involucrados = []
        scenario_key = "zona_segura"

        # --- EVALUACIÓN DE RIESGOS (orden: mayor → menor severidad) ---

        # 1. CRITICAL: persona con bbox solapado o muy cerca de vehículo
        for person in persons:
            for vehicle in vehicles:
                # Zona de peligro expandida alrededor del vehículo
                danger_zone = _bbox_expand(
                    vehicle["bbox"], self.DANGER_ZONE_FACTOR, img_w, img_h
                )
                overlap = _bboxes_overlap(person["bbox"], danger_zone)
                dist    = _center_dist_norm(person["bbox"], vehicle["bbox"], img_w, img_h)

                if overlap or dist < 0.08:   # IoU real o muy cerca
                    scenario_key = "atropello_inminente"
                    involucrados = [person, vehicle]
                    return self._build_result(scenario_key, involucrados, stats)

        # 2. WARNING: persona cercana (dentro del umbral de proximidad)
        for person in persons:
            for vehicle in vehicles:
                dist = _center_dist_norm(person["bbox"], vehicle["bbox"], img_w, img_h)
                if dist < self.PROXIMITY_THRESHOLD:
                    scenario_key = "persona_cerca_vehiculo"
                    involucrados = [person, vehicle]
                    break
            if scenario_key != "zona_segura":
                break

        # 3. WARNING: ciclista entre vehículos grandes
        if scenario_key == "zona_segura":
            for bicycle in bicycles:
                for lv in large_vehs:
                    dist = _center_dist_norm(bicycle["bbox"], lv["bbox"], img_w, img_h)
                    if dist < self.PROXIMITY_THRESHOLD * 1.3:
                        scenario_key = "ciclista_entre_vehiculos"
                        involucrados = [bicycle, lv]
                        break
                if scenario_key != "zona_segura":
                    break

        # 4. WARNING: ciclista con vehículos presentes
        if scenario_key == "zona_segura" and bicycles and vehicles:
            scenario_key = "ciclista_sin_proteccion"
            involucrados = bicycles[:1] + vehicles[:1]

        # 5. WARNING: peatones con mucho tráfico
        if scenario_key == "zona_segura" and persons and len(vehicles) >= 2:
            scenario_key = "trafico_denso_peaton"
            involucrados = persons[:1] + vehicles[:2]

        # 6. WARNING: personas presentes con al menos un vehículo
        if scenario_key == "zona_segura" and persons and vehicles:
            scenario_key = "peaton_en_calzada"
            involucrados = persons[:1] + vehicles[:1]

        # 7. Iluminación baja — combina con escenario existente si hay tráfico
        if lighting and lighting["is_dark"]:
            if vehicles and scenario_key in ("zona_segura", "peaton_en_calzada",
                                             "trafico_denso_peaton"):
                # Oscuridad + vehículos = CRITICAL educativo
                scenario_key = "zona_oscura_con_trafico"
                involucrados = (persons[:1] if persons else []) + vehicles[:1]
            elif scenario_key == "zona_segura":
                # Oscuridad sin tráfico = WARNING
                scenario_key = "zona_oscura"
                involucrados = []
            # Si ya hay un escenario CRITICAL activo, solo se añade info de iluminación en stats

        return self._build_result(scenario_key, involucrados, stats)

    # ------------------------------------------------------------------
    def _build_result(
        self,
        scenario_key: str,
        involucrados: List[Dict],
        stats: Dict,
    ) -> Dict[str, Any]:
        sc = SCENARIOS[scenario_key]
        return {
            "risk_level":       sc["risk_level"],
            "scenario_key":     scenario_key,
            "titulo":           sc["titulo"],
            "explicacion":      sc["explicacion"],
            "leccion":          sc["leccion"],
            "consejo_rapido":   sc["consejo_rapido"],
            "color_alerta":     sc["color_alerta"],
            "detecciones_clave": [
                {"clase": d["class_name"], "confianza": d["confidence"], "bbox": d["bbox"]}
                for d in involucrados
            ],
            "stats": stats,
        }
