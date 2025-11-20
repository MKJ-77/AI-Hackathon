import time
from collections import deque

maze = [
    [0, 0, 0, 1, 0, 2, 0],
    [1, 1, 0, 1, 0, 1, 0],
    [0, 2, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2],
]

agents = [[0, 0], [4, 0]]
victims = [[0, 5], [2, 1], [4, 6]]
rescued = []
ROWS, COLS = len(maze), len(maze[0])

def bfs(start, goal):
    q = deque([(start, [])])
    seen = {tuple(start)}
    while q:
        (x, y), path = q.popleft()
        if [x, y] == goal:
            return path + [[x, y]]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if (0 <= nx < ROWS and 0 <= ny < COLS 
                and maze[nx][ny] != 1 and (nx, ny) not in seen):
                seen.add((nx, ny))
                q.append(([nx, ny], path + [[x, y]]))
    return []

while victims:
    # Assign: for every agent, if idle, assign closest victim
    targets = []
    for agent in agents:
        if victims:
            best_path = None
            best_v = None
            best_len = float('inf')
            for v in victims:
                path = bfs(agent, v)
                if path and len(path) < best_len:
                    best_v = v
                    best_path = path
                    best_len = len(path)
            if best_path:  # Move agent one step along path
                move = best_path[1] if len(best_path)>1 else best_path[0]
                agent[0], agent[1] = move
        targets.append(list(agent)) # for display
    
    # Check for rescues
    for agent in agents:
        for v in victims:
            if agent == v:
                rescued.append(v)
    victims = [v for v in victims if v not in rescued]
                
    # Display
    for r in range(ROWS):
        row = ""
        for c in range(COLS):
            ch = "."
            if maze[r][c]==1:
                ch = "#"
            elif [r,c] in victims:
                ch = "V"
            elif [r,c] in rescued:
                ch = "R"
            for agent in agents:
                if [r,c]==agent:
                    ch = "A"
            row += ch + " "
        print(row)
    print()
    time.sleep(0.6)

print("All victims rescued!")
