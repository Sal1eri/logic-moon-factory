import heapq
from typing import Dict, List, Optional

from lunar_path.environment import Coord, LunarMap, heuristic, neighbors, transition_cost, transition_energy


def astar(lunar: LunarMap, risk_aware: bool = True, battery_constrained: bool = False) -> Optional[List[Coord]]:
    start, goal = lunar.start, lunar.goal
    open_heap = [(heuristic(start, goal), 0.0, start)]
    came_from: Dict[Coord, Coord] = {}
    g_score = {start: 0.0}
    energy_score = {start: 0.0}
    visited = set()

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for nxt, move in neighbors(current, lunar):
            step_cost = transition_cost(lunar, current, nxt, move) if risk_aware else move
            step_energy = transition_energy(lunar, current, nxt, move)
            next_energy = energy_score[current] + step_energy
            if battery_constrained and next_energy > lunar.battery_capacity:
                continue
            tentative = current_g + step_cost
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                energy_score[nxt] = next_energy
                heapq.heappush(open_heap, (tentative + heuristic(nxt, goal), tentative, nxt))
    return None
