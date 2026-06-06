from typing import Dict, List, Optional

from lunar_path.environment import Coord, LunarMap, heuristic, neighbors


def depth_first_search(lunar: LunarMap, max_expansions: int = 20000) -> Optional[List[Coord]]:
    start, goal = lunar.start, lunar.goal
    stack = [start]
    came_from: Dict[Coord, Coord] = {}
    visited = {start}
    expansions = 0

    while stack and expansions < max_expansions:
        current = stack.pop()
        expansions += 1
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        ordered = sorted((nxt for nxt, _ in neighbors(current, lunar)), key=lambda p: heuristic(p, goal), reverse=True)
        for nxt in ordered:
            if nxt in visited:
                continue
            visited.add(nxt)
            came_from[nxt] = current
            stack.append(nxt)
    return None

