"""
Test rápido del servidor LUCY.
Simula lo que haría el Blueprint de UE5.

Uso (con el servidor ya corriendo):
    python server/test_server.py
"""
import requests
import json

BASE = "http://localhost:8765"


def post(url, payload):
    r = requests.post(url, json=payload)
    if not r.ok or not r.text.strip():
        print(f"  [ERROR] status={r.status_code}")
        print(f"  [ERROR] body={r.text[:500]}")
        return None
    try:
        return r.json()
    except Exception as e:
        print(f"  [ERROR] JSON parse failed: {e}")
        print(f"  [ERROR] raw response: {r.text[:500]}")
        return None

# ── 1. Health check ───────────────────────────────────────────────────────────
print("=== Test 1: Health ===")
r = requests.get(f"{BASE}/health")
print(r.json())

# ── 2. Status ─────────────────────────────────────────────────────────────────
print("\n=== Test 2: Status ===")
r = requests.get(f"{BASE}/status")
print(json.dumps(r.json(), indent=2))

# ── 3. Escena segura ──────────────────────────────────────────────────────────
print("\n=== Test 3: Escena segura (parque) ===")
payload = {
    "detections": [
        {"class_id": 0, "bbox": [200, 300, 260, 480], "confidence": 0.92},
        {"class_id": 0, "bbox": [380, 280, 440, 460], "confidence": 0.88},
    ],
    "brightness": 180.0,
    "weapon_near": False,
    "reset_sequence": True
}
res = post(f"{BASE}/analyze", payload)
if res:
    print(f"  LUCY decide:  {res['lucy_action'].upper()}")
    print(f"  Comportamiento: {res['behavior']} ({res['behavior_conf']*100:.0f}%)")
    print(f"  Riesgo:         {res['risk_level']} ({res['risk_conf']*100:.0f}%)")
    print(f"  HUD título:     {res['hud']['title']}")

# ── 4. Escena peligrosa ───────────────────────────────────────────────────────
print("\n=== Test 4: Escena crítica (camión + personas) ===")
payload = {
    "detections": [
        {"class_id": 0, "bbox": [295, 300, 365, 500], "confidence": 0.97},
        {"class_id": 7, "bbox": [170, 170, 560, 555], "confidence": 0.95},
        {"class_id": 2, "bbox": [440, 280, 610, 450], "confidence": 0.81},
    ],
    "brightness": 60.0,
    "weapon_near": False,
    "reset_sequence": True
}
res = post(f"{BASE}/analyze", payload)
if res:
    print(f"  LUCY decide:  {res['lucy_action'].upper()}")
    print(f"  Comportamiento: {res['behavior']}")
    print(f"  Riesgo:         {res['risk_level']}")
    print(f"  HUD mensaje:    {res['hud']['message']}")

# ── 5. Arma detectada ─────────────────────────────────────────────────────────
print("\n=== Test 5: Arma detectada ===")
payload = {
    "detections": [
        {"class_id": 0, "bbox": [300, 350, 360, 530], "confidence": 0.94},
        {"class_id": 0, "bbox": [340, 345, 400, 525], "confidence": 0.91},
    ],
    "brightness": 120.0,
    "weapon_near": True,
    "reset_sequence": True
}
res = post(f"{BASE}/analyze", payload)
if res:
    print(f"  LUCY decide:  {res['lucy_action'].upper()}")
    print(f"  HUD color:    R={res['hud']['color']['r']} G={res['hud']['color']['g']} B={res['hud']['color']['b']}")
    print(f"  HUD mensaje:  {res['hud']['message']}")

# ── 6. Simular 30 frames seguidos (LSTM se activa) ───────────────────────────
print("\n=== Test 6: 30 frames — LSTM se activa ===")
payload_base = {
    "detections": [
        {"class_id": 0, "bbox": [300, 350, 360, 530], "confidence": 0.94},
    ],
    "brightness": 128.0,
    "weapon_near": False,
    "reset_sequence": True
}
res = post(f"{BASE}/analyze", payload_base)
if res is None:
    print("  Falló el primer frame")
else:
    payload_base["reset_sequence"] = False
    for i in range(29):
        res = post(f"{BASE}/analyze", payload_base)
    if res:
        print(f"  Buffer lleno: {res['behavior_ready']}")
        print(f"  LUCY decide:  {res['lucy_action'].upper()}")
        print(f"  Comportamiento detectado: {res['behavior']} ({res['behavior_conf']*100:.0f}%)")

print("\n✓ Servidor listo para UE5.")
