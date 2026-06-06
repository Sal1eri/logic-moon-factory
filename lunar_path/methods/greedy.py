from typing import List, Optional

from lunar_path.environment import Coord, LunarMap, heuristic, neighbors, transition_cost, transition_energy


def greedy_best_first(lunar: LunarMap, risk_weight: float = 0.35, max_steps: int = 1800) -> Optional[List[Coord]]:
    pos = lunar.start
    path: List[Coord] = [pos]
    visited = {pos}
    energy_used = 0.0
    for _ in range(max_steps):
        if pos == lunar.goal:
            return path
        options = []
        for nxt, move in neighbors(pos, lunar):
            revisit_penalty = 6.0 if nxt in visited else 0.0
            score = heuristic(nxt, lunar.goal) + risk_weight * transition_cost(lunar, pos, nxt, move) + revisit_penalty
            options.append((score, nxt))
        if not options:
            return None
        options.sort(key=lambda item: item[0])
        pos = options[0][1]
        prev = path[-1]
        dy = pos[0] - prev[0]
        dx = pos[1] - prev[1]
        move = 2 ** 0.5 if dy and dx else 1.0
        energy_used += transition_energy(lunar, prev, pos, move)
        path.append(pos)
        if energy_used > lunar.battery_capacity:
            return path
        if pos == lunar.goal:
            return path
        if pos in visited:
            return path
        visited.add(pos)
    return path
