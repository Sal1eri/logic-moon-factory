from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN, PPO

from lunar_path.environment import ACTIONS, Coord, LunarMap, LunarPathEnv, heuristic, transition_cost, transition_energy


class TrainingLogger:
    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        self.rewards = []
        self.current_reward = 0.0
        self.episode = 0

    def __call__(self, locals_, globals_):
        rewards = locals_.get("rewards")
        dones = locals_.get("dones")
        if rewards is not None:
            self.current_reward += float(np.asarray(rewards).reshape(-1)[0])
        if dones is not None and bool(np.asarray(dones).reshape(-1)[0]):
            self.episode += 1
            self.rewards.append({"episode": self.episode, "algorithm": self.algorithm, "reward": self.current_reward})
            self.current_reward = 0.0
        return True


def train_deep_rl(lunar: LunarMap, algorithm: str, timesteps: int, seed: int, device: str) -> Tuple[object, pd.DataFrame]:
    env = LunarPathEnv(lunar, seed=seed)
    logger = TrainingLogger(algorithm)
    policy_kwargs = dict(net_arch=[128, 128])

    if algorithm == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=8e-4,
            buffer_size=50000,
            learning_starts=600,
            batch_size=256,
            gamma=0.94,
            train_freq=4,
            target_update_interval=750,
            exploration_fraction=0.45,
            exploration_final_eps=0.04,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=device,
        )
    elif algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=4e-4,
            n_steps=512,
            batch_size=256,
            n_epochs=5,
            gamma=0.94,
            gae_lambda=0.90,
            ent_coef=0.02,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=device,
        )
    else:
        raise ValueError(algorithm)

    model.learn(total_timesteps=timesteps, callback=logger)
    logs = pd.DataFrame(logger.rewards)
    if logs.empty:
        logs = pd.DataFrame([{"episode": 1, "algorithm": algorithm, "reward": np.nan}])
    return model, logs


def deep_rl_policy_path(lunar: LunarMap, model, max_steps: Optional[int] = None) -> Optional[List[Coord]]:
    env = LunarPathEnv(lunar)
    obs, _ = env.reset()
    path = [lunar.start]
    seen = {lunar.start}
    if max_steps is None:
        max_steps = lunar.size * lunar.size
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = choose_safe_action(lunar, env.pos, obs, model, int(action), seen, env.energy_used)
        if action is None:
            return path
        obs, _, done, truncated, info = env.step(action)
        path.append(env.pos)
        if info.get("battery_failed", False):
            return path
        if done:
            return path
        if truncated or env.pos in seen:
            return None
        seen.add(env.pos)
    return None


def choose_safe_action(lunar: LunarMap, pos: Coord, obs: np.ndarray, model, proposed: int, seen, energy_used: float) -> Optional[int]:
    ranked_actions = rank_actions(lunar, obs, model, proposed)
    valid = []
    for action in ranked_actions:
        dy, dx = ACTIONS[action]
        nxt = (pos[0] + dy, pos[1] + dx)
        if not (0 <= nxt[0] < lunar.size and 0 <= nxt[1] < lunar.size):
            continue
        if lunar.obstacle[nxt] or nxt in seen:
            continue
        move = 2 ** 0.5 if dy and dx else 1.0
        next_energy = energy_used + transition_energy(lunar, pos, nxt, move)
        if next_energy > lunar.battery_capacity:
            continue
        score = heuristic(nxt, lunar.goal) + 0.4 * transition_cost(lunar, pos, nxt, move)
        valid.append((score, action))
    if not valid:
        return None
    return min(valid, key=lambda item: item[0])[1]


def rank_actions(lunar: LunarMap, obs: np.ndarray, model, proposed: int):
    if isinstance(model, DQN):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device).reshape(1, -1)
            q_values = model.q_net(obs_tensor).detach().cpu().numpy().reshape(-1)
        return [int(a) for a in np.argsort(q_values)[::-1]]
    actions = [proposed]
    actions.extend([a for a in range(len(ACTIONS)) if a != proposed])
    return actions
