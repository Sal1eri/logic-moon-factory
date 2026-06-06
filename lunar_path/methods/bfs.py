from collections import deque
from typing import Dict, List, Optional

from lunar_path.environment import Coord, LunarMap, neighbors


def breadth_first_search(lunar: LunarMap) -> Optional[List[Coord]]:
    start, goal = lunar.start, lunar.goal
    queue = deque([start])
    came_from: Dict[Coord, Coord] = {}
    visited = {start}

    while queue:
        current = queue.popleft()
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for nxt, _ in neighbors(current, lunar):
            if nxt in visited:
                continue
            visited.add(nxt)
            came_from[nxt] = current
            queue.append(nxt)
    return None

