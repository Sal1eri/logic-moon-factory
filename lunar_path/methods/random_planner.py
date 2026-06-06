import random
from typing import List, Optional

from lunar_path.environment import Coord, LunarMap, neighbors, transition_energy


def random_walk(lunar: LunarMap, seed: int = 0, max_steps: int = 1800, restarts: int = 40) -> Optional[List[Coord]]:
    rng = random.Random(seed)
    best_path = None
    best_dist = float("inf")
    for _ in range(restarts):
        pos = lunar.start
        path = [pos]
        visited = {pos}
        energy_used = 0.0
        for _ in range(max_steps):
            if pos == lunar.goal:
                return path
            options = list(neighbors(pos, lunar))
            if not options:
                break
            unvisited = [item for item in options if item[0] not in visited]
            nxt, move = rng.choice(unvisited or options)
            energy_used += transition_energy(lunar, pos, nxt, move)
            pos = nxt
            path.append(pos)
            if energy_used > lunar.battery_capacity:
                break
            visited.add(pos)
        dist = abs(pos[0] - lunar.goal[0]) + abs(pos[1] - lunar.goal[1])
        if dist < best_dist:
            best_dist = dist
            best_path = path
    return best_path
