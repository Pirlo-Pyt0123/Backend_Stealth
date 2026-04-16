"""
Script de descarga de datasets desde Roboflow Universe.
Módulo Seguridad Ciudadana + Riesgo Vial (perspectiva peatón).
"""
from roboflow import Roboflow
import os

RF_KEY = "JgFTjbkzIdfMONazBcSd"
BASE   = os.path.dirname(os.path.abspath(__file__))

rf = Roboflow(api_key=RF_KEY)

datasets = [
    # ── Módulo Seguridad Ciudadana ──────────────────────────────────────
    {
        "workspace": "fd",
        "project":   "fight-detection",
        "version":   1,
        "fmt":       "yolov8",
        "dest":      "fight_detection",
        "desc":      "Detección de peleas/violencia",
    },
    {
        "workspace": "fd",
        "project":   "fightdetection",
        "version":   1,
        "fmt":       "yolov8",
        "dest":      "fight_detection_v2",
        "desc":      "Fight detection v2 - 431 imgs, 6 clases",
    },
    # ── Módulo Riesgo Vial ─────────────────────────────────────────────
    {
        "workspace": "training-data-kgqsn",
        "project":   "pedestrian-detection-v6aln",
        "version":   1,
        "fmt":       "yolov8",
        "dest":      "pedestrian_detection",
        "desc":      "Detección de peatones en entornos urbanos",
    },
    {
        "workspace": "lynkeus03",
        "project":   "vehicle-detection-by9xs",
        "version":   1,
        "fmt":       "yolov8",
        "dest":      "vehicle_detection",
        "desc":      "9211 imágenes de vehículos (car, bus, truck, motorbike)",
    },
]

def download_all():
    results = []
    for ds in datasets:
        dest = os.path.join(BASE, ds["dest"])
        if os.path.exists(dest) and os.listdir(dest):
            print(f"[SKIP] {ds['dest']} ya existe")
            results.append((ds['dest'], 'skipped'))
            continue
        print(f"\n[DOWNLOAD] {ds['desc']}")
        print(f"  workspace={ds['workspace']}  project={ds['project']}  v{ds['version']}")
        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            version = project.version(ds["version"])
            version.download(ds["fmt"], location=dest)
            print(f"  ✓ guardado en: {dest}")
            results.append((ds['dest'], 'ok'))
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append((ds['dest'], f'error: {e}'))

    print("\n─── Resumen ───")
    for name, status in results:
        icon = "✓" if status in ('ok', 'skipped') else "✗"
        print(f"  {icon} {name}: {status}")

if __name__ == "__main__":
    download_all()
