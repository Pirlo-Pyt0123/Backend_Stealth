"""
Entrenamiento del Agente RL (PPO) de LUCY.

El agente aprende a integrar todas las señales de percepción
y decidir el nivel de alerta correcto mediante trial-and-error.

No tiene reglas de cuándo alertar.
No tiene umbrales programados.
Descubre la política óptima solo, por reinforcement learning.

Uso:
    cd c:/Users/LENOVO/Documents/Stealth
    python -m training.train_rl_agent
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pickle
from pathlib import Path
from collections import deque

from models.rl_environment import (
    RiskScenarioEnv, ActorCritic, PPOTrainer,
    ACTIONS, N_ACTIONS, STATE_DIM
)

# ── Configuración ─────────────────────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
TOTAL_TIMESTEPS = 500_000
N_STEPS         = 2048
N_EPOCHS        = 10
BATCH_SIZE      = 64
LR              = 3e-4
MODELS_DIR      = Path("models")

print(f"[LUCY] Entrenando PPO Risk Agent en: {DEVICE}")
print(f"       Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"       State dim: {STATE_DIM}  |  Actions: {ACTIONS}")


def evaluate_agent(agent: ActorCritic, env: RiskScenarioEnv, n_episodes: int = 20):
    """Evalúa el agente sin exploración (deterministic)."""
    agent.eval()
    total_correct = 0
    total_steps   = 0
    rewards_hist  = []

    for _ in range(n_episodes):
        state  = env.reset()
        ep_rew = 0.0
        for _ in range(env.n_steps):
            action, _, _ = agent.act(state, deterministic=True)
            state, reward, done, info = env.step(action)
            ep_rew += reward
            total_correct += int(info["correct"])
            total_steps   += 1
            if done:
                break
        rewards_hist.append(ep_rew)

    acc = total_correct / total_steps * 100
    return float(np.mean(rewards_hist)), acc


def train():
    env   = RiskScenarioEnv(n_steps=N_STEPS, seed=42)
    agent = ActorCritic(state_dim=STATE_DIM, n_actions=N_ACTIONS)
    trainer = PPOTrainer(
        agent      = agent,
        env        = env,
        lr         = LR,
        n_steps    = N_STEPS,
        n_epochs   = N_EPOCHS,
        batch_size = BATCH_SIZE,
        device     = DEVICE,
    )

    history = {"update": [], "mean_reward": [], "accuracy": [], "loss": []}
    best_acc = 0.0
    updates  = TOTAL_TIMESTEPS // N_STEPS

    print(f"\n[1] Iniciando entrenamiento PPO — {updates} updates")
    print(f"    Parámetros del agente: {sum(p.numel() for p in agent.parameters()):,}\n")

    for upd in range(1, updates + 1):
        # Recolectar experiencia
        states, actions, log_probs, rewards, values, dones, last_s = \
            trainer._collect_rollout()

        # Calcular ventajas GAE
        advantages, returns = trainer._compute_gae(rewards, values, dones, last_s)

        # Actualizar política
        loss = trainer.update(states, actions, log_probs, returns, advantages)

        # Logging cada 10 updates
        if upd % 10 == 0:
            mean_r = float(rewards.mean())
            # Accuracy rápida (comparar acción greedy vs GT en batch)
            agent.eval()
            with torch.no_grad():
                s_t       = torch.FloatTensor(states).to(DEVICE)
                logits, _ = agent.forward(s_t)
                preds  = logits.argmax(dim=-1).cpu().numpy()
                # Recalcular GT usando las recompensas del batch
                # (aproximación: reward >0 → acertó)
                correct = (rewards > 0).sum()
                acc     = correct / len(rewards) * 100

            history["update"].append(upd)
            history["mean_reward"].append(mean_r)
            history["accuracy"].append(float(acc))
            history["loss"].append(loss)

            print(f"  Update {upd:4d}/{updates} | "
                  f"reward {mean_r:+6.2f} | "
                  f"acc {acc:.1f}% | "
                  f"loss {loss:.4f} | "
                  f"steps {upd*N_STEPS:,}")

            if float(acc) > best_acc:
                best_acc = float(acc)
                torch.save(agent.state_dict(), MODELS_DIR / "rl_agent_best.pth")

    # Evaluación final
    print(f"\n[2] Evaluación final del agente PPO...")
    agent.load_state_dict(torch.load(MODELS_DIR / "rl_agent_best.pth"))
    eval_env = RiskScenarioEnv(n_steps=N_STEPS, seed=999)
    mean_r, acc = evaluate_agent(agent, eval_env, n_episodes=50)
    print(f"    Mean reward: {mean_r:.2f}  |  Accuracy: {acc:.1f}%")

    # Análisis por clase
    print(f"\n[3] Análisis por clase de riesgo:")
    agent.eval()
    class_correct = {a: 0 for a in ACTIONS}
    class_total   = {a: 0 for a in ACTIONS}
    state = eval_env.reset()
    for _ in range(5000):
        action, _, _ = agent.act(state, deterministic=True)
        state, _, done, info = eval_env.step(action)
        gt_name = ACTIONS[info["gt_label"]]
        class_total[gt_name] += 1
        if info["correct"]:
            class_correct[gt_name] += 1
        if done:
            state = eval_env.reset()
    for cls in ACTIONS:
        if class_total[cls] > 0:
            cls_acc = class_correct[cls] / class_total[cls] * 100
            print(f"    {cls:<10}: {cls_acc:.1f}% ({class_correct[cls]}/{class_total[cls]})")

    # Guardar
    with open(MODELS_DIR / "rl_agent_history.pkl", "wb") as f:
        pickle.dump(history, f)

    agent.export_onnx(str(MODELS_DIR / "rl_agent.onnx"))

    print(f"\n[LUCY] PPO Agent entrenado. Mejor accuracy: {best_acc:.1f}%")
    print(f"       Guardado en: models/rl_agent_best.pth")
    return agent, history


if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    train()
