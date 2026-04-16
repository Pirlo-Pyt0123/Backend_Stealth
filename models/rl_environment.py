"""
RL Environment + PPO Agent para LUCY.

El agente aprende a evaluar riesgo combinando las señales del
BehaviorLSTM y el SpatialRiskTransformer.

Sin reglas. Sin umbrales. El agente DESCUBRE la política óptima
de alerta mediante trial-and-error.

Entorno:
  - Estado  : vector de features de la escena (percepción de LUCY)
  - Acción  : [SAFE=0, WARNING=1, CRITICAL=2]
  - Reward  :
      +15  acertó nivel exacto
      +5   acertó entre WARNING y CRITICAL (error de un nivel)
      -20  dijo SAFE cuando era WARNING
      -50  dijo SAFE cuando era CRITICAL  ← el peor error para LUCY
      -30  dijo CRITICAL cuando era SAFE  ← demasiado nervioso
      -10  dijo WARNING cuando era CRITICAL

Arquitectura PPO:
  Estado → Actor (policy) → distribución sobre acciones
         → Crítico (value) → V(s) para reducir varianza

El Actor y el Crítico comparten un encoder profundo (shared backbone).
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import random

# ── Constantes ────────────────────────────────────────────────────────────────
ACTIONS      = ["SAFE", "WARNING", "CRITICAL"]
N_ACTIONS    = len(ACTIONS)        # 3
STATE_DIM    = 32                  # dimensión del vector de estado
GAMMA        = 0.99                # descuento temporal
GAE_LAMBDA   = 0.95               # lambda para Generalized Advantage Estimation
CLIP_EPS     = 0.2                 # epsilon de clipping PPO
ENTROPY_COEF = 0.01                # coeficiente de entropía (exploración)
VF_COEF      = 0.5                 # coeficiente de value function loss


# ─────────────────────────────────────────────────────────────────────────────
# REWARD MATRIX
# reward_matrix[predicción][ground_truth]
REWARD_MATRIX = {
    0: {0: +15, 1: -20, 2: -50},   # predijo SAFE
    1: {0:  -5, 1: +15, 2: -10},   # predijo WARNING
    2: {0: -30, 1:  +5, 2: +15},   # predijo CRITICAL
}


# ─────────────────────────────────────────────────────────────────────────────
class RiskScenarioEnv:
    """
    Entorno de simulación para entrenar el agente RL de LUCY.

    Genera episodios de escenas con niveles de riesgo conocidos
    y evalúa las decisiones del agente contra el ground truth.

    Uso:
        env   = RiskScenarioEnv()
        state = env.reset()
        for _ in range(200):
            action        = agent.act(state)
            state, reward, done, info = env.step(action)
    """

    def __init__(self, n_steps: int = 200, seed: int = 42):
        self.n_steps     = n_steps
        self.rng         = np.random.default_rng(seed)
        self._step       = 0
        self._state      = np.zeros(STATE_DIM, dtype=np.float32)
        self._gt_label   = 0
        self._episode_rewards: List[float] = []

    # ── Observación ──────────────────────────────────────────────────────────
    def _generate_scenario(self) -> Tuple[np.ndarray, int]:
        """
        Genera un escenario de riesgo aleatorio con su label ground truth.

        El estado incluye features ricas que el agente aprende a interpretar:
          [0]   n_persons norm
          [1]   n_vehicles norm
          [2]   n_weapons
          [3]   min_person_vehicle_dist
          [4]   person_vehicle_overlap
          [5]   min_person_person_dist
          [6]   person_person_overlap
          [7]   encirclement_signal
          [8]   loitering_signal (del LSTM)
          [9]   following_signal (del LSTM)
          [10]  violence_signal (del LSTM)
          [11]  brightness_norm
          [12]  is_dark
          [13]  time_of_day_norm  (0=noche, 1=día)
          [14]  lstm_behavior_probs × 6   (probs del BehaviorLSTM)
          [20]  spatial_safe_prob
          [21]  spatial_warning_prob
          [22]  spatial_critical_prob
          [23]  combined_threat_score
          [24]  person_isolated_dark
          [25]  weapon_near_person
          [26]  large_vehicle_near
          [27]  multiple_fast_vehicles
          [28]  pedestrian_on_road
          [29]  running_near_traffic
          [30]  scene_complexity
          [31]  time_since_last_alert_norm
        """
        rng = self.rng
        label = int(rng.integers(0, 3))    # SAFE=0, WARNING=1, CRITICAL=2

        s = np.zeros(STATE_DIM, dtype=np.float32)

        if label == 0:    # SAFE
            s[0]  = rng.uniform(0.0, 0.3)   # pocas personas
            s[1]  = rng.uniform(0.0, 0.2)   # pocos vehículos
            s[2]  = 0.0
            s[3]  = rng.uniform(0.5, 1.0)   # lejos
            s[4]  = 0.0
            s[5]  = rng.uniform(0.4, 1.0)
            s[11] = rng.uniform(0.6, 1.0)   # buena luz
            s[12] = 0.0
            s[13] = rng.uniform(0.5, 1.0)
            s[14] = rng.uniform(0.7, 1.0)   # prob normal alta
            s[20] = rng.uniform(0.6, 1.0)   # spatial SAFE alto
            s[21] = rng.uniform(0.0, 0.3)
            s[22] = rng.uniform(0.0, 0.1)
            s[23] = rng.uniform(0.0, 0.2)

        elif label == 1:  # WARNING
            s[0]  = rng.uniform(0.2, 0.6)
            s[1]  = rng.uniform(0.2, 0.6)
            s[2]  = 0.0
            s[3]  = rng.uniform(0.15, 0.35)  # cerca pero no overlap
            s[4]  = 0.0
            s[5]  = rng.uniform(0.1, 0.4)
            s[8]  = rng.uniform(0.3, 0.8)    # loitering moderado
            s[11] = rng.uniform(0.3, 0.7)
            s[12] = float(rng.random() > 0.5)
            s[13] = rng.uniform(0.2, 0.8)
            s[14] = rng.uniform(0.2, 0.5)
            s[20] = rng.uniform(0.1, 0.4)
            s[21] = rng.uniform(0.4, 0.8)    # spatial WARNING alto
            s[22] = rng.uniform(0.0, 0.3)
            s[23] = rng.uniform(0.3, 0.6)
            s[25] = 0.0
            s[26] = float(rng.random() > 0.4)
            s[28] = float(rng.random() > 0.4)

        else:             # CRITICAL
            s[0]  = rng.uniform(0.3, 1.0)
            s[1]  = rng.uniform(0.2, 0.8)
            s[2]  = float(rng.random() > 0.5)   # arma posible
            s[3]  = rng.uniform(0.0, 0.10)       # muy cerca / overlap
            s[4]  = float(rng.random() > 0.4)    # overlap real
            s[5]  = rng.uniform(0.0, 0.12)
            s[6]  = float(rng.random() > 0.5)
            s[7]  = float(rng.random() > 0.4)    # encirclement
            s[10] = rng.uniform(0.5, 1.0)        # violencia
            s[11] = rng.uniform(0.0, 0.4)        # oscuro
            s[12] = float(rng.random() > 0.3)
            s[13] = rng.uniform(0.0, 0.4)        # noche
            s[14] = rng.uniform(0.0, 0.3)
            s[20] = rng.uniform(0.0, 0.15)
            s[21] = rng.uniform(0.1, 0.4)
            s[22] = rng.uniform(0.5, 1.0)        # spatial CRITICAL alto
            s[23] = rng.uniform(0.6, 1.0)
            s[24] = float(rng.random() > 0.4)    # persona sola + oscuro
            s[25] = float(rng.random() > 0.5)    # arma cerca de persona
            s[29] = float(rng.random() > 0.4)    # corriendo cerca de tráfico

        # Ruido realista
        s += rng.normal(0, 0.03, STATE_DIM).astype(np.float32)
        s  = np.clip(s, 0.0, 1.0)

        return s, label

    # ── Gym API ───────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        self._step = 0
        self._episode_rewards = []
        self._state, self._gt_label = self._generate_scenario()
        return self._state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        reward = float(REWARD_MATRIX[action][self._gt_label])
        self._episode_rewards.append(reward)
        self._step += 1
        done = self._step >= self.n_steps
        self._state, self._gt_label = self._generate_scenario()
        info = {
            "gt_label":    self._gt_label,
            "action_name": ACTIONS[action],
            "correct":     action == self._gt_label,
        }
        return self._state.copy(), reward, done, info

    @property
    def episode_return(self) -> float:
        return sum(self._episode_rewards)


# ─────────────────────────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    """
    Red Actor-Crítico compartida para PPO.

    Backbone compartido (aprende representaciones del estado de riesgo)
    ↓
    Actor  → distribución de probabilidad sobre acciones (policy)
    Crítico → estimación del valor del estado V(s)
    """

    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS):
        super().__init__()

        # Backbone compartido
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_actions),
        )

        # Crítico (value function)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Última capa del actor con ganancia menor → exploración inicial
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B, state_dim)
        Retorna: (action_logits (B, n_actions), value (B, 1))
        """
        feat   = self.backbone(x)
        logits = self.actor(feat)
        value  = self.critic(feat)
        return logits, value

    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Selecciona acción dado el estado.
        deterministic=True para evaluación (sin exploración).
        Retorna (action, log_prob, value)
        """
        device = next(self.parameters()).device
        x      = torch.FloatTensor(state).unsqueeze(0).to(device)
        logits, value = self.forward(x)
        dist   = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def evaluate(
        self,
        states:  torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evalúa acciones pasadas para el update PPO.
        Retorna: log_probs, values, entropy
        """
        logits, values = self.forward(states)
        dist      = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values.squeeze(-1), entropy

    def export_onnx(self, path: str):
        """Exporta a ONNX para UE5 NNE."""
        try:
            self.eval()
            dummy = torch.zeros(1, STATE_DIM)
            torch.onnx.export(
                self, dummy, path,
                input_names  = ["state"],
                output_names = ["action_logits", "value"],
                dynamic_axes = {"state": {0: "batch"}, "action_logits": {0: "batch"}},
                opset_version = 17,
            )
            print(f"[LUCY] PPO ActorCritic → ONNX: {path}")
        except Exception as e:
            print(f"[LUCY] ONNX export omitido ({e}). El .pth sigue disponible.")


# ─────────────────────────────────────────────────────────────────────────────
class PPOTrainer:
    """
    Implementación PPO (Proximal Policy Optimization) para LUCY.

    Algoritmo:
      1. Recolectar T pasos de experiencia con la policy actual
      2. Calcular ventajas con GAE
      3. Actualizar policy K épocas minimizando loss clipeado
      4. Repetir

    PPO garantiza actualizaciones conservadoras → entrenamiento estable.
    """

    def __init__(
        self,
        agent:         ActorCritic,
        env:           RiskScenarioEnv,
        lr:            float = 3e-4,
        n_steps:       int   = 2048,
        n_epochs:      int   = 10,
        batch_size:    int   = 64,
        device:        str   = "cpu",
    ):
        self.agent      = agent.to(device)
        self.env        = env
        self.optimizer  = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
        self.n_steps    = n_steps
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.device     = device
        self.scheduler  = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=500
        )

    def _collect_rollout(self):
        """Recolecta n_steps de experiencia."""
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        state = self.env.reset()

        for _ in range(self.n_steps):
            action, lp, val = self.agent.act(state)
            next_state, reward, done, _ = self.env.step(action)
            states.append(state); actions.append(action)
            log_probs.append(lp); rewards.append(reward)
            values.append(val);   dones.append(done)
            state = self.env.reset() if done else next_state

        return (
            np.array(states), np.array(actions),
            np.array(log_probs), np.array(rewards),
            np.array(values), np.array(dones), state
        )

    def _compute_gae(self, rewards, values, dones, last_state):
        """Generalized Advantage Estimation — reduce varianza del gradiente."""
        with torch.no_grad():
            _, last_val, _ = self.agent.act(last_state)
        advantages = np.zeros_like(rewards)
        last_gae   = 0.0
        for t in reversed(range(len(rewards))):
            nxt_val = last_val if t == len(rewards)-1 else values[t+1]
            delta   = rewards[t] + GAMMA * nxt_val * (1-dones[t]) - values[t]
            advantages[t] = last_gae = delta + GAMMA * GAE_LAMBDA * (1-dones[t]) * last_gae
        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, old_log_probs, returns, advantages):
        """Un ciclo de actualización PPO con K épocas."""
        dev = self.device
        S = torch.FloatTensor(states).to(dev)
        A = torch.LongTensor(actions).to(dev)
        O = torch.FloatTensor(old_log_probs).to(dev)
        R = torch.FloatTensor(returns).to(dev)
        Adv = torch.FloatTensor(advantages).to(dev)
        Adv = (Adv - Adv.mean()) / (Adv.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.n_epochs):
            idx = torch.randperm(len(S))
            for start in range(0, len(S), self.batch_size):
                b   = idx[start:start + self.batch_size]
                lp, val, ent = self.agent.evaluate(S[b], A[b])
                ratio    = torch.exp(lp - O[b])
                surr1    = ratio * Adv[b]
                surr2    = ratio.clamp(1-CLIP_EPS, 1+CLIP_EPS) * Adv[b]
                pi_loss  = -torch.min(surr1, surr2).mean()
                vf_loss  = F.mse_loss(val, R[b])
                ent_loss = -ent.mean()
                loss     = pi_loss + VF_COEF * vf_loss + ENTROPY_COEF * ent_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()
        self.scheduler.step()
        return total_loss

    def train(self, total_timesteps: int = 200_000, log_every: int = 10):
        """
        Entrena el agente PPO por total_timesteps pasos.
        Devuelve historial de métricas para graficar.
        """
        history = {"rewards": [], "accuracy": [], "loss": []}
        updates = total_timesteps // self.n_steps
        step    = 0

        for upd in range(1, updates + 1):
            states, actions, log_probs, rewards, values, dones, last_s = \
                self._collect_rollout()
            advantages, returns = self._compute_gae(rewards, values, dones, last_s)
            loss = self.update(states, actions, log_probs, returns, advantages)
            step += self.n_steps

            if upd % log_every == 0:
                mean_r = float(rewards.mean())
                acc    = float((actions == self.env._gt_label).mean()) * 100
                history["rewards"].append(mean_r)
                history["accuracy"].append(acc)
                history["loss"].append(loss)
                print(f"  Update {upd:4d}/{updates} | "
                      f"steps {step:7d} | "
                      f"reward {mean_r:+6.2f} | "
                      f"loss {loss:.4f}")

        return history
