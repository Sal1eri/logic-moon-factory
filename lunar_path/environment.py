import math
from dataclasses import dataclass, replace
from typing import Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


Coord = Tuple[int, int]

ACTIONS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    size: int
    seed: int
    crater_count: int
    rock_density: float
    regolith_patches: int
    shadow_patches: int
    comm_stations: Tuple[Coord, ...]
    slope_scale: float
    start: Coord
    goal: Coord
    rl_timesteps: int
    battery_capacity: float


@dataclass
class LunarMap:
    size: int
    elevation: np.ndarray
    slope: np.ndarray
    obstacle: np.ndarray
    crater_risk: np.ndarray
    regolith: np.ndarray
    illumination: np.ndarray
    communication: np.ndarray
    cost: np.ndarray
    start: Coord
    goal: Coord
    battery_capacity: float


def heuristic(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def smooth_noise(rng: np.random.Generator, size: int, passes: int = 5) -> np.ndarray:
    data = rng.normal(0, 1, (size, size))
    for _ in range(passes):
        padded = np.pad(data, 1, mode="edge")
        data = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0
    data -= data.min()
    return data / (data.max() + 1e-9)


def add_circular_patch(arr: np.ndarray, center: Coord, radius: float, value: float) -> None:
    yy, xx = np.indices(arr.shape)
    mask = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2) <= radius
    arr[mask] = np.maximum(arr[mask], value)


def generate_lunar_map(scenario: Scenario) -> LunarMap:
    rng = np.random.default_rng(scenario.seed)
    n = scenario.size
    elevation = smooth_noise(rng, n, passes=8) * scenario.slope_scale
    obstacle = np.zeros((n, n), dtype=bool)
    crater_risk = np.zeros((n, n), dtype=float)
    regolith = np.zeros((n, n), dtype=float)
    illumination = np.ones((n, n), dtype=float)

    yy, xx = np.indices((n, n))
    for _ in range(scenario.crater_count):
        center = (int(rng.integers(6, n - 6)), int(rng.integers(6, n - 6)))
        radius = float(rng.uniform(3.0, 8.0))
        dist = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
        inner = dist <= radius * 0.43
        rim = (dist > radius * 0.43) & (dist <= radius)
        obstacle[inner] = True
        crater_risk[rim] = np.maximum(crater_risk[rim], 0.85)
        crater_risk[inner] = 1.0
        elevation += np.where(rim, 0.25 * scenario.slope_scale, 0.0)
        elevation -= np.where(inner, 0.18 * scenario.slope_scale, 0.0)

    obstacle |= rng.random((n, n)) < scenario.rock_density

    for _ in range(scenario.regolith_patches):
        center = (int(rng.integers(4, n - 4)), int(rng.integers(4, n - 4)))
        add_circular_patch(regolith, center, float(rng.uniform(4, 10)), float(rng.uniform(0.45, 0.85)))

    for _ in range(scenario.shadow_patches):
        center = (int(rng.integers(4, n - 4)), int(rng.integers(4, n - 4)))
        radius = float(rng.uniform(5, 12))
        dist = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
        shadow = np.clip(1.0 - dist / radius, 0, 1)
        illumination = np.minimum(illumination, 1.0 - 0.9 * shadow)

    gy, gx = np.gradient(elevation)
    slope = np.sqrt(gx**2 + gy**2)
    slope = np.clip(slope / (np.percentile(slope, 98) + 1e-9), 0, 1)
    obstacle |= slope > 0.93

    communication = np.zeros((n, n), dtype=float)
    for station in scenario.comm_stations:
        dist = np.sqrt((yy - station[0]) ** 2 + (xx - station[1]) ** 2)
        communication = np.maximum(communication, np.clip(1.0 - dist / (n * 0.42), 0, 1))

    for point in (scenario.start, scenario.goal):
        obstacle[point] = False
        crater_risk[point] = 0
        regolith[point] = 0
        illumination[point] = 1

    cost = (
        1.0
        + 2.8 * slope
        + 2.2 * crater_risk
        + 1.7 * regolith
        + 1.6 * (1.0 - illumination)
        + 1.3 * (1.0 - communication)
    )
    cost[obstacle] = np.inf
    return LunarMap(
        n,
        elevation,
        slope,
        obstacle,
        crater_risk,
        regolith,
        illumination,
        communication,
        cost,
        scenario.start,
        scenario.goal,
        scenario.battery_capacity,
    )


def neighbors(pos: Coord, lunar: LunarMap):
    y, x = pos
    for dy, dx in ACTIONS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < lunar.size and 0 <= nx < lunar.size and not lunar.obstacle[ny, nx]:
            move = math.sqrt(2) if dy and dx else 1.0
            yield (ny, nx), move


def transition_energy(lunar: LunarMap, src: Coord, dst: Coord, move_distance: float) -> float:
    elevation_gain = max(0.0, float(lunar.elevation[dst] - lunar.elevation[src]))
    terrain_multiplier = (
        1.0
        + 2.2 * float(lunar.slope[dst])
        + 1.7 * float(lunar.regolith[dst])
        + 1.1 * elevation_gain
        + 0.8 * float(1.0 - lunar.illumination[dst])
    )
    return move_distance * terrain_multiplier


def transition_cost(lunar: LunarMap, src: Coord, dst: Coord, move_distance: float) -> float:
    energy = transition_energy(lunar, src, dst, move_distance)
    terrain_risk = 2.2 * float(lunar.crater_risk[dst]) + 1.0 * float(lunar.slope[dst])
    communication_risk = 1.3 * float(1.0 - lunar.communication[dst])
    return energy + terrain_risk + communication_risk


def execute_path_with_battery(lunar: LunarMap, path):
    if not path:
        return path
    executed = [path[0]]
    energy_used = 0.0
    for src, dst in zip(path[:-1], path[1:]):
        dy = dst[0] - src[0]
        dx = dst[1] - src[1]
        move = math.sqrt(2) if dy and dx else 1.0
        energy_used += transition_energy(lunar, src, dst, move)
        executed.append(dst)
        if energy_used > lunar.battery_capacity:
            break
    return executed


def step_env(lunar: LunarMap, state: Coord, action_idx: int):
    dy, dx = ACTIONS[action_idx]
    ny, nx = state[0] + dy, state[1] + dx
    if not (0 <= ny < lunar.size and 0 <= nx < lunar.size) or lunar.obstacle[ny, nx]:
        return state, -18.0, False
    nxt = (ny, nx)
    move = math.sqrt(2) if dy and dx else 1.0
    reward = -transition_cost(lunar, state, nxt, move)
    done = nxt == lunar.goal
    if done:
        reward += 180.0
    return nxt, reward, done


def observation_for_state(lunar: LunarMap, pos: Coord, energy_used: float) -> np.ndarray:
    y, x = pos
    gy, gx = lunar.goal
    n = lunar.size - 1
    global_features = [
        2 * y / n - 1,
        2 * x / n - 1,
        2 * gy / n - 1,
        2 * gx / n - 1,
        np.clip((gy - y) / lunar.size, -1, 1),
        np.clip((gx - x) / lunar.size, -1, 1),
        2 * lunar.slope[y, x] - 1,
        2 * lunar.crater_risk[y, x] - 1,
        2 * lunar.regolith[y, x] - 1,
        2 * lunar.illumination[y, x] - 1,
        2 * lunar.communication[y, x] - 1,
        np.clip(lunar.cost[y, x] / 8.0, 0, 1) * 2 - 1,
        np.clip(heuristic(pos, lunar.goal) / (lunar.size * np.sqrt(2)), 0, 1) * 2 - 1,
        np.clip(1.0 - energy_used / lunar.battery_capacity, 0, 1) * 2 - 1,
    ]
    local_features = []
    for dy, dx in ACTIONS:
        nxt = (y + dy, x + dx)
        valid = 0 <= nxt[0] < lunar.size and 0 <= nxt[1] < lunar.size and not lunar.obstacle[nxt]
        if valid:
            move = math.sqrt(2) if dy and dx else 1.0
            step_cost = transition_cost(lunar, pos, nxt, move)
            local_features.extend([1.0, np.clip(step_cost / 12.0, 0, 1) * 2 - 1])
        else:
            local_features.extend([-1.0, 1.0])
    return np.array(global_features + local_features, dtype=np.float32)


class LunarPathEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, lunar: LunarMap, seed: int = 0):
        super().__init__()
        self.lunar = lunar
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(30,), dtype=np.float32)
        self.max_steps = lunar.size * lunar.size // 2
        self.pos = lunar.start
        self.steps = 0
        self.energy_used = 0.0

    def _obs(self) -> np.ndarray:
        return observation_for_state(self.lunar, self.pos, self.energy_used)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.lunar.start
        self.steps = 0
        self.energy_used = 0.0
        return self._obs(), {}

    def step(self, action):
        old = self.pos
        old_dist = heuristic(old, self.lunar.goal)
        nxt, base_reward, done = step_env(self.lunar, old, int(action))
        self.steps += 1
        if nxt != old:
            dy = nxt[0] - old[0]
            dx = nxt[1] - old[1]
            move = math.sqrt(2) if dy and dx else 1.0
            self.energy_used += transition_energy(self.lunar, old, nxt, move)
        new_dist = heuristic(nxt, self.lunar.goal)
        old_phi = -old_dist
        new_phi = -new_dist
        potential_shaping = 4.0 * (0.94 * new_phi - old_phi)
        reward = base_reward + potential_shaping
        if nxt == old and not done:
            reward -= 7.0
        battery_failed = self.energy_used > self.lunar.battery_capacity
        if battery_failed:
            reward -= 180.0
        self.pos = nxt
        truncated = self.steps >= self.max_steps
        if done:
            reward += 120.0
        terminated = done or battery_failed
        return self._obs(), float(reward), terminated, truncated, {"battery_failed": battery_failed, "energy_used": self.energy_used}


class RandomizedLunarPathEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenarios: Sequence[Scenario], seed: int = 0):
        super().__init__()
        self.scenarios = list(scenarios)
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(30,), dtype=np.float32)
        self.inner = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        base = self.scenarios[int(self.rng.integers(0, len(self.scenarios)))]
        sampled_seed = int(base.seed + self.rng.integers(10000, 1000000))
        scenario = replace(base, seed=sampled_seed)
        self.inner = LunarPathEnv(generate_lunar_map(scenario), seed=sampled_seed)
        return self.inner.reset()

    def step(self, action):
        return self.inner.step(action)
