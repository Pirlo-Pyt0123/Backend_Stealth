"""
TEST 01 — Narrative Generator (sin GPU, sin modelos)
====================================================
Verifica que build_narrative() produce texto correcto para cada escenario
posible. Corre instantáneamente — no necesita modelos ni GPU.

Si quieres escuchar la voz al mismo tiempo, usa --speak:
    python scripts/test_01_narrative.py --speak
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from modules.voice_narrator import build_narrative, VoiceNarrator

# ── Escenarios de prueba ───────────────────────────────────────────────────────

SCENARIOS = [
    {
        "nombre": "Zona despejada",
        "result": {
            "risk_level": "SAFE",
            "confidence": 0.97,
            "tl_states": [],
            "tl_override": False,
            "stats": {"personas": 0, "vehiculos": 0, "bicicletas": 0,
                      "grandes": 0, "semaforos": 0},
        }
    },
    {
        "nombre": "Peatón solo, sin vehículos",
        "result": {
            "risk_level": "SAFE",
            "confidence": 0.91,
            "tl_states": ["GREEN"],
            "tl_override": False,
            "stats": {"personas": 1, "vehiculos": 0, "bicicletas": 0,
                      "grandes": 0, "semaforos": 1, "is_dark": False},
        }
    },
    {
        "nombre": "Semáforo amarillo con tráfico",
        "result": {
            "risk_level": "WARNING",
            "confidence": 0.78,
            "tl_states": ["YELLOW"],
            "tl_override": False,
            "stats": {"personas": 2, "vehiculos": 3, "bicicletas": 0,
                      "grandes": 0, "semaforos": 1, "is_dark": False},
        }
    },
    {
        "nombre": "Warning - vehículos sin bici ni grande",
        "result": {
            "risk_level": "WARNING",
            "confidence": 0.83,
            "tl_states": [],
            "tl_override": False,
            "stats": {"personas": 1, "vehiculos": 2, "bicicletas": 0,
                      "grandes": 0, "semaforos": 0},
        }
    },
    {
        "nombre": "Ciclista entre vehículos",
        "result": {
            "risk_level": "WARNING",
            "confidence": 0.75,
            "tl_states": [],
            "tl_override": False,
            "stats": {"personas": 0, "vehiculos": 3, "bicicletas": 2,
                      "grandes": 0, "semaforos": 0},
        }
    },
    {
        "nombre": "Semáforo rojo — override activo",
        "result": {
            "risk_level": "WARNING",
            "confidence": 0.88,
            "tl_states": ["RED"],
            "tl_override": True,
            "stats": {"personas": 2, "vehiculos": 0, "bicicletas": 0,
                      "grandes": 0, "semaforos": 1, "is_dark": False},
        }
    },
    {
        "nombre": "CRÍTICO — peatones + vehículos",
        "result": {
            "risk_level": "CRITICAL",
            "confidence": 0.94,
            "tl_states": [],
            "tl_override": False,
            "stats": {"personas": 3, "vehiculos": 2, "bicicletas": 0,
                      "grandes": 0, "semaforos": 0},
        }
    },
    {
        "nombre": "CRÍTICO — semáforo rojo + cruce activo",
        "result": {
            "risk_level": "CRITICAL",
            "confidence": 0.97,
            "tl_states": ["RED"],
            "tl_override": False,
            "stats": {"personas": 2, "vehiculos": 3, "bicicletas": 0,
                      "grandes": 0, "semaforos": 1, "is_dark": False},
        }
    },
    {
        "nombre": "CRÍTICO — vehículo grande + múltiples",
        "result": {
            "risk_level": "CRITICAL",
            "confidence": 0.92,
            "tl_states": [],
            "tl_override": False,
            "stats": {"personas": 1, "vehiculos": 4, "bicicletas": 0,
                      "grandes": 1, "semaforos": 0},
        }
    },
]

FIGHT_SCENARIOS = [
    {
        "nombre": "PELEA detectada (confianza media)",
        "result": {
            "risk_level": "WARNING",
            "confidence": 0.72,
            "tl_states": [],
            "tl_override": False,
            "stats": {"personas": 2, "vehiculos": 0, "bicicletas": 0,
                      "grandes": 0, "semaforos": 0},
            "fight_detected":   True,
            "fight_confidence": 0.67,
        }
    },
    {
        "nombre": "PELEA detectada (alta confianza)",
        "result": {
            "risk_level": "CRITICAL",
            "confidence": 0.90,
            "tl_states": [],
            "tl_override": False,
            "stats": {"personas": 3, "vehiculos": 0, "bicicletas": 0,
                      "grandes": 0, "semaforos": 0},
            "fight_detected":   True,
            "fight_confidence": 0.88,
        }
    },
    {
        "nombre": "Sin pelea (fight_detected=False no interfiere)",
        "result": {
            "risk_level": "SAFE",
            "confidence": 0.95,
            "tl_states": [],
            "tl_override": False,
            "stats": {"personas": 1, "vehiculos": 0, "bicicletas": 0,
                      "grandes": 0, "semaforos": 0},
            "fight_detected":   False,
            "fight_confidence": 0.12,
        }
    },
]

COLORS = {
    "SAFE":     "\033[92m",   # verde
    "WARNING":  "\033[93m",   # amarillo
    "CRITICAL": "\033[91m",   # rojo
    "FIGHT":    "\033[95m",   # magenta
    "RESET":    "\033[0m",
}


def run(speak: bool = False):
    import time

    narrator = VoiceNarrator() if speak else None

    print("\n" + "=" * 65)
    print("  LUCY VoiceNarrator — Test de Narrativa Contextual")
    print("=" * 65)

    if narrator:
        print("\n  Esperando edge-tts...", end="", flush=True)
        narrator.wait_until_ready(timeout=30)
        print(f" listo! ({narrator.status['mode']})\n")

    all_scenarios = SCENARIOS + FIGHT_SCENARIOS

    for i, s in enumerate(all_scenarios, 1):
        fight_active = s["result"].get("fight_detected", False)
        level   = "FIGHT" if fight_active else s["result"]["risk_level"]
        conf    = s["result"]["confidence"]
        text    = build_narrative(s["result"])
        color   = COLORS.get(level, "")
        reset   = COLORS["RESET"]

        print(f"\n[{i:02d}] {s['nombre']}")
        print(f"     Nivel : {color}{level}{reset}  (confianza: {conf:.0%})")
        print(f"     LUCY  : {color}\"{text}\"{reset}")

        if narrator:
            narrator.speak(s["result"])
            # Forzar reset de estado para que hable cada escenario aunque el nivel no cambie
            narrator._last_traffic_level = None
            narrator._last_fight_state   = False
            narrator._last_spoken        = 0.0
            time.sleep(7)

    print("\n" + "=" * 65)
    print(f"  {len(all_scenarios)} escenarios verificados ({len(SCENARIOS)} tráfico + {len(FIGHT_SCENARIOS)} pelea)")
    if not speak:
        print("  Tip: agrega --speak para escuchar la voz")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speak", action="store_true",
                        help="Reproducir voz para cada escenario")
    args = parser.parse_args()
    run(speak=args.speak)
