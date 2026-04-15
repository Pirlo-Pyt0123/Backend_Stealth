"""
train_scenario_classifier.py — Genera datos sintéticos, entrena clasificadores ML
de escenarios y guarda los modelos en models/.

Arquitectura:
    Detecciones RT-DETR → FeatureExtractor → GradientBoostingClassifier → scenario + probs

Los datos de entrenamiento se generan sintéticamente colocando bboxes de forma
parametrizada y etiquetando con los motores de reglas originales como "profesor".

Uso (desde la raíz del proyecto):
    python -m training.train_scenario_classifier

Salida:
    models/traffic_clf.pkl
    models/security_clf.pkl
"""

import sys
import os
# Asegura que la raíz del proyecto esté en sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from engines.feature_extractor import (
    extract_traffic_features,
    extract_security_features,
    TRAFFIC_FEATURE_NAMES,
    SECURITY_FEATURE_NAMES,
    CLASS_PERSON, CLASS_BICYCLE, CLASS_CAR, CLASS_MOTORCYCLE,
    CLASS_BUS, CLASS_TRUCK, CLASS_KNIFE, CLASS_SCISSORS,
    CLASS_BACKPACK, CLASS_HANDBAG,
)

# Importamos los motores con su lógica de reglas para generar etiquetas
from engines.traffic_risk_engine   import TrafficRiskEngine
from engines.suspicious_behavior_engine import SuspiciousBehaviorEngine

RNG      = np.random.default_rng(42)
IMG_W    = 640
IMG_H    = 640
IMG_SHAPE = (IMG_H, IMG_W, 3)
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 9: "traffic light", 11: "stop sign",
    24: "backpack", 26: "handbag", 43: "knife", 76: "scissors",
}


# ── Helpers de generación de bboxes ───────────────────────────────────────

def _det(class_id, bbox, conf=0.90):
    return {
        "class_id":   class_id,
        "class_name": COCO_NAMES.get(class_id, f"cls_{class_id}"),
        "confidence": round(float(conf), 3),
        "bbox":       bbox,
    }


def _bbox(cx_f, cy_f, w_f, h_f, jitter=0.04):
    """Genera bbox centrado en (cx_f, cy_f) con tamaño w_f×h_f (fracción de imagen)."""
    cx = float(np.clip(cx_f + RNG.uniform(-jitter, jitter), 0.05, 0.95)) * IMG_W
    cy = float(np.clip(cy_f + RNG.uniform(-jitter, jitter), 0.05, 0.95)) * IMG_H
    w  = w_f * float(RNG.uniform(0.8, 1.2)) * IMG_W
    h  = h_f * float(RNG.uniform(0.8, 1.2)) * IMG_H
    x1 = max(0, int(cx - w / 2))
    y1 = max(0, int(cy - h / 2))
    x2 = min(IMG_W - 1, int(cx + w / 2))
    y2 = min(IMG_H - 1, int(cy + h / 2))
    return [x1, y1, x2, y2]


def _person(cx, cy, jitter=0.04):
    return _det(CLASS_PERSON, _bbox(cx, cy, 0.06, 0.18, jitter))


def _vehicle(cx, cy, large=False, jitter=0.04):
    cls = RNG.choice([CLASS_BUS, CLASS_TRUCK]) if large \
          else RNG.choice([CLASS_CAR, CLASS_MOTORCYCLE, CLASS_BUS, CLASS_TRUCK])
    w = 0.28 if cls in (CLASS_BUS, CLASS_TRUCK) else 0.18
    return _det(int(cls), _bbox(cx, cy, w, 0.14, jitter))


def _bicycle(cx, cy):
    return _det(CLASS_BICYCLE, _bbox(cx, cy, 0.07, 0.12))


def _weapon(cx, cy):
    cls = int(RNG.choice([CLASS_KNIFE, CLASS_SCISSORS]))
    return _det(cls, _bbox(cx, cy, 0.04, 0.08, jitter=0.02))


def _fake_image(brightness):
    """Imagen 32x32 uniforme — suficiente para análisis de brillo, mínima RAM."""
    b = int(np.clip(brightness, 0, 255))
    return np.full((32, 32, 3), b, dtype=np.uint8)


def _rand_pos():
    return float(RNG.uniform(0.1, 0.9)), float(RNG.uniform(0.1, 0.9))


# ── Generadores de escenarios de tráfico ──────────────────────────────────

def _gen_traffic_batch(n=7000):
    """
    Genera n escenarios por tipo de situación vial.
    Devuelve lista de (detections, fake_image).
    """
    samples = []

    # 1. zona_segura — sin vehículos
    for _ in range(n):
        dets = [_person(*_rand_pos()) for _ in range(int(RNG.integers(0, 4)))]
        samples.append((dets, _fake_image(RNG.uniform(100, 255))))

    # 2. atropello_inminente — persona encima del vehículo
    for _ in range(n):
        vx, vy = _rand_pos()
        v = _vehicle(vx, vy, jitter=0.01)
        p = _person(vx + float(RNG.uniform(-0.02, 0.02)),
                    vy + float(RNG.uniform(-0.02, 0.02)), jitter=0.01)
        extras = [_vehicle(*_rand_pos()) for _ in range(int(RNG.integers(0, 3)))]
        samples.append(([p, v] + extras, _fake_image(RNG.uniform(100, 255))))

    # 3. persona_cerca_vehiculo — cercana pero sin overlap
    for _ in range(n):
        vx, vy = _rand_pos()
        offset = float(RNG.uniform(0.09, 0.17))
        sign   = float(RNG.choice([-1.0, 1.0]))
        px     = float(np.clip(vx + sign * offset, 0.05, 0.95))
        py     = float(np.clip(vy + RNG.uniform(-0.05, 0.05), 0.05, 0.95))
        samples.append(([_person(px, py), _vehicle(vx, vy)],
                        _fake_image(RNG.uniform(100, 255))))

    # 4. ciclista_entre_vehiculos — bici cerca de vehículo grande
    for _ in range(n):
        lx, ly = _rand_pos()
        lv     = _det(int(RNG.choice([CLASS_BUS, CLASS_TRUCK])),
                      _bbox(lx, ly, 0.28, 0.16))
        offset = float(RNG.uniform(0.12, 0.22))
        bx     = float(np.clip(lx + float(RNG.choice([-1.0, 1.0])) * offset, 0.05, 0.95))
        extras = [_vehicle(*_rand_pos()) for _ in range(int(RNG.integers(0, 2)))]
        samples.append(([_bicycle(bx, ly), lv] + extras,
                        _fake_image(RNG.uniform(100, 255))))

    # 5. ciclista_sin_proteccion — bici con vehículos normales lejos de grandes
    for _ in range(n):
        bx, by = float(RNG.uniform(0.05, 0.45)), _rand_pos()[1]
        vehs   = [_det(int(RNG.choice([CLASS_CAR, CLASS_MOTORCYCLE])),
                       _bbox(float(RNG.uniform(0.5, 0.95)), _rand_pos()[1], 0.16, 0.12))
                  for _ in range(int(RNG.integers(1, 4)))]
        samples.append(([_bicycle(bx, by)] + vehs,
                        _fake_image(RNG.uniform(100, 255))))

    # 6. trafico_denso_peaton — múltiples personas y vehículos
    for _ in range(n):
        persons  = [_person(*_rand_pos()) for _ in range(int(RNG.integers(1, 4)))]
        vehicles = [_vehicle(*_rand_pos()) for _ in range(int(RNG.integers(2, 5)))]
        samples.append((persons + vehicles, _fake_image(RNG.uniform(100, 255))))

    # 7. zona_oscura — sin tráfico, oscuro
    for _ in range(n):
        dets = [_person(*_rand_pos()) for _ in range(int(RNG.integers(0, 3)))]
        samples.append((dets, _fake_image(RNG.uniform(20, 85))))

    # 8. zona_oscura_con_trafico — oscuro + vehículos
    for _ in range(n):
        persons  = [_person(*_rand_pos()) for _ in range(int(RNG.integers(0, 3)))]
        vehicles = [_vehicle(*_rand_pos()) for _ in range(int(RNG.integers(1, 4)))]
        samples.append((persons + vehicles, _fake_image(RNG.uniform(20, 85))))

    return samples


# ── Generadores de escenarios de seguridad ────────────────────────────────

def _gen_security_batch(n=7000):
    """Genera n escenarios por tipo de situación de seguridad."""
    samples = []

    # 1. situacion_normal — personas dispersas, sin amenazas
    for _ in range(n):
        dets = [_person(*_rand_pos()) for _ in range(int(RNG.integers(0, 3)))]
        samples.append((dets, _fake_image(RNG.uniform(100, 255)), 0, 0.0))

    # 2. arma_detectada — arma cerca de persona
    for _ in range(n):
        px, py = _rand_pos()
        wx     = float(np.clip(px + float(RNG.uniform(-0.08, 0.08)), 0.05, 0.95))
        wy     = float(np.clip(py + float(RNG.uniform(-0.08, 0.08)), 0.05, 0.95))
        extras = [_person(*_rand_pos()) for _ in range(int(RNG.integers(0, 2)))]
        samples.append(([_person(px, py), _weapon(wx, wy)] + extras,
                        _fake_image(RNG.uniform(100, 255)), 0, 0.0))

    # 3. confrontacion — 2 personas con bbox solapado y muy cerca
    for _ in range(n):
        cx, cy = _rand_pos()
        p1 = _person(cx - 0.02, cy, jitter=0.01)
        p2 = _person(cx + 0.02, cy, jitter=0.01)
        extras = [_person(*_rand_pos()) for _ in range(int(RNG.integers(0, 2)))]
        samples.append(([p1, p2] + extras,
                        _fake_image(RNG.uniform(100, 255)), 0, 0.0))

    # 4. encercamiento — 1 target rodeado por 3+ en distintos cuadrantes
    for _ in range(n):
        cx, cy = 0.5, 0.5
        target = _person(cx, cy, jitter=0.02)
        dist   = float(RNG.uniform(0.12, 0.25))
        surrounders = [
            _person(cx + dist, cy, jitter=0.03),   # E
            _person(cx - dist, cy, jitter=0.03),   # W
            _person(cx, cy + dist, jitter=0.03),   # S
            _person(cx, cy - dist, jitter=0.03),   # N
        ][:int(RNG.integers(3, 5))]
        samples.append(([target] + surrounders,
                        _fake_image(RNG.uniform(100, 255)), 0, 0.0))

    # 5. merodeo — simulamos con loitering_count > 0 en feature vector
    for _ in range(n):
        dets          = [_person(*_rand_pos()) for _ in range(int(RNG.integers(1, 3)))]
        loiter_count  = int(RNG.integers(1, 4))
        loiter_secs   = float(RNG.uniform(5.0, 30.0))
        samples.append((dets, _fake_image(RNG.uniform(100, 255)),
                        loiter_count, loiter_secs))

    # 6. zona_oscura_persona — oscuro con personas
    for _ in range(n):
        dets = [_person(*_rand_pos()) for _ in range(int(RNG.integers(1, 4)))]
        samples.append((dets, _fake_image(RNG.uniform(20, 85)), 0, 0.0))

    # 7. grupo_sospechoso — 4+ personas
    for _ in range(n):
        dets = [_person(*_rand_pos()) for _ in range(int(RNG.integers(4, 8)))]
        samples.append((dets, _fake_image(RNG.uniform(100, 255)), 0, 0.0))

    return samples


# ── Construcción de datasets ───────────────────────────────────────────────

def build_traffic_dataset():
    print("Generando datos de tráfico...")
    engine  = TrafficRiskEngine()
    samples = _gen_traffic_batch(n=7000)

    X, y = [], []
    for dets, img in samples:
        feats  = extract_traffic_features(dets, IMG_SHAPE, img)
        result = engine._analyze_rules(dets, IMG_SHAPE, img)
        X.append(feats)
        y.append(result["scenario_key"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"  Total muestras: {len(y)}")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"    {cls}: {cnt}")
    return X, y


def build_security_dataset():
    print("Generando datos de seguridad...")
    engine  = SuspiciousBehaviorEngine()
    samples = _gen_security_batch(n=7000)

    X, y = [], []
    for dets, img, loiter_count, loiter_secs in samples:
        feats = extract_security_features(dets, IMG_SHAPE, img)
        # Inyectamos info de merodeo directamente en el vector
        feats[9]  = float(loiter_count)
        feats[10] = min(loiter_secs / 10.0, 1.0)

        result = engine._analyze_rules(
            dets, IMG_SHAPE, img,
            loitering_count=loiter_count,
            max_loiter_seconds=loiter_secs,
        )
        X.append(feats)
        y.append(result["scenario_key"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"  Total muestras: {len(y)}")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"    {cls}: {cnt}")
    return X, y


# ── Entrenamiento ──────────────────────────────────────────────────────────

def train_classifier(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"\nEntrenando {name}...")
    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\n--- Reporte {name} ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Feature importances
    importances = sorted(
        zip(clf.feature_names_in_ if hasattr(clf, "feature_names_in_")
            else range(X.shape[1]),
            clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("Top-5 features más importantes:")
    for feat, imp in importances[:5]:
        print(f"  {feat}: {imp:.4f}")

    return clf


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Tráfico ──
    X_t, y_t = build_traffic_dataset()
    clf_traffic = train_classifier(X_t, y_t, "TrafficClassifier")
    path_t = os.path.join(MODELS_DIR, "traffic_clf.pkl")
    with open(path_t, "wb") as f:
        pickle.dump(clf_traffic, f)
    print(f"\nModelo guardado: {path_t}")

    # ── Seguridad ──
    X_s, y_s = build_security_dataset()
    clf_security = train_classifier(X_s, y_s, "SecurityClassifier")
    path_s = os.path.join(MODELS_DIR, "security_clf.pkl")
    with open(path_s, "wb") as f:
        pickle.dump(clf_security, f)
    print(f"Modelo guardado: {path_s}")

    print("\nEntrenamiento completado. Reinicia el servidor para cargar los modelos.")


if __name__ == "__main__":
    main()
