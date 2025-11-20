# dual_maze_navigators.py
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# %matplotlib inline (Add this at the top of collab)


# ---- Config ----
ROWS, COLS = 20, 28
WALL_PROB = 0.18
NUM_KEYS = 8
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---- Utility grid functions ----
def make_grid(rows, cols, wall_prob):
    grid = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            if random.random() < wall_prob:
                grid[r, c] = 1  # wall
    return grid

def in_bounds(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

dirs = [(1,0),(-1,0),(0,1),(0,-1)]

def bfs_shortest_path(grid, start, target):
    if start == target:
        return [start]
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == target:
            break
        r, c = cur
        for dr, dc in dirs:
            nr, nc = r+dr, c+dc
            if not in_bounds(nr, nc): continue
            if grid[nr, nc] == 1: continue
            if (nr, nc) not in prev:
                prev[(nr, nc)] = cur
                q.append((nr, nc))
    if target not in prev:
        return None
    # reconstruct
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path

def nearest_key_bfs(grid, start, key_set):
    # returns key coordinate and path to it (shortest)
    q = deque([start])
    visited = {start}
    prev = {start: None}
    if start in key_set:
        return start, [start]
    while q:
        cur = q.popleft()
        r, c = cur
        for dr, dc in dirs:
            nr, nc = r+dr, c+dc
            if not in_bounds(nr, nc): continue
            if grid[nr, nc] == 1: continue
            nxt = (nr, nc)
            if nxt in visited: continue
            visited.add(nxt)
            prev[nxt] = cur
            if nxt in key_set:
                # reconstruct
                path = []
                cur2 = nxt
                while cur2 is not None:
                    path.append(cur2)
                    cur2 = prev[cur2]
                path.reverse()
                return nxt, path
            q.append(nxt)
    return None, None

# ---- Maze setup ----
grid = make_grid(ROWS, COLS, WALL_PROB)

# Ensure start areas are open
startA = (1,1)
startB = (ROWS-2, COLS-2)
for (r,c) in [startA, startB]:
    grid[r,c] = 0

# Place keys in random empty spots
empty = [(r,c) for r in range(ROWS) for c in range(COLS) if grid[r,c]==0 and (r,c) not in (startA,startB)]
random.shuffle(empty)
keys = set(empty[:NUM_KEYS])

# Agents state
agents = {
    'A': {'pos': startA, 'path': [], 'collected': [], 'color': 0.2},
    'B': {'pos': startB, 'path': [], 'collected': [], 'color': 0.8},
}

# Shared resources
uncollected = set(keys)
reserved = {}  # key_coord -> agent_id

# For visualization: record trails
trailA = []
trailB = []

# ---- Simulation loop ----
max_steps = 2000
step = 0
history = []  # record positions for animation

while uncollected and step < max_steps:
    step += 1
    # For each agent, if no path, pick nearest unreserved key
    for aid in ['A', 'B']:
        agent = agents[aid]
        # if path exists and still valid, keep moving along it
        if agent['path'] and len(agent['path'])>1:
            continue  # already has plan
        # choose nearest key not reserved (or reserved by self)
        available_keys = [k for k in uncollected if reserved.get(k) in (None, aid)]
        if not available_keys:
            # no available keys — agent stays idle or re-evaluates later
            agent['path'] = []
            continue
        key_coord, path = nearest_key_bfs(grid, agent['pos'], set(available_keys))
        if key_coord is None:
            # unreachable keys; just idle
            agent['path'] = []
            continue
        # reserve and set path
        reserved[key_coord] = aid
        agent['path'] = path  # path includes current pos possibly

    # Propose moves (next cell in each agent's path if any)
    proposes = {}
    for aid in ['A','B']:
        agent = agents[aid]
        if agent['path'] and len(agent['path'])>=2:
            proposes[aid] = agent['path'][1]
        else:
            proposes[aid] = agent['pos']  # staying

    # Simple collision resolution: if both propose same cell, break tie by shorter path-to-target
    if proposes['A'] == proposes['B'] and proposes['A'] != agents['A']['pos']:
        # compute remaining distances (if any)
        distA = len(agents['A']['path']) if agents['A']['path'] else 9999
        distB = len(agents['B']['path']) if agents['B']['path'] else 9999
        if distA <= distB:
            # A moves, B waits (remove B's next step)
            moves = {'A': proposes['A'], 'B': agents['B']['pos']}
        else:
            moves = {'A': agents['A']['pos'], 'B': proposes['B']}
    else:
        moves = {'A': proposes['A'], 'B': proposes['B']}

    # Apply moves: advance along path if moved
    for aid in ['A','B']:
        agent = agents[aid]
        newpos = moves[aid]
        # If different, pop first step from path
        if newpos != agent['pos']:
            # remove steps until newpos (should be the next)
            if agent['path'] and agent['path'][1] == newpos:
                agent['path'].pop(0)
                agent['pos'] = newpos
            else:
                # unexpected; recompute path next loop
                agent['path'] = []
                agent['pos'] = newpos
        # record trail
        if aid == 'A':
            trailA.append(agent['pos'])
        else:
            trailB.append(agent['pos'])

        # key pickup
        if agent['pos'] in uncollected:
            uncollected.remove(agent['pos'])
            agent['collected'].append(agent['pos'])
            # clear reservation
            if agent['pos'] in reserved:
                del reserved[agent['pos']]

    # If some reserved keys became unreserved (other agent gave up), clean mapping
    # Remove reservations that point to unreachable or already collected keys
    to_del = [k for k,v in reserved.items() if k not in uncollected]
    for k in to_del:
        del reserved[k]

    history.append((agents['A']['pos'], agents['B']['pos'], set(uncollected)))
    # safety: if both stuck with no path and keys remain unreachable -> break
    if all((not agents[aid]['path'] for aid in agents)) and uncollected:
        # try global reassignment: remove reservations, allow both to pick any
        reserved.clear()
        # small chance to random-walk a bit to escape trapped positions
        for aid in agents:
            r,c = agents[aid]['pos']
            for dr,dc in random.sample(dirs, len(dirs)):
                nr, nc = r+dr, c+dc
                if in_bounds(nr,nc) and grid[nr,nc]==0:
                    agents[aid]['pos'] = (nr,nc)
                    break

# ---- Visualization via matplotlib ----
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Dual Maze Navigators: A (left) and B (right)")

# prepare base image: walls black, empty white
canvas = np.ones((ROWS, COLS, 3))
for r in range(ROWS):
    for c in range(COLS):
        if grid[r,c] == 1:
            canvas[r,c] = [0.0, 0.0, 0.0]  # wall black
        else:
            canvas[r,c] = [1.0, 1.0, 1.0]  # free white

# overlay keys (use red dots)
key_map = np.zeros((ROWS, COLS, 3))
for (kr,kc) in keys:
    key_map[kr,kc] = [1.0, 0.0, 0.0]

img = ax.imshow(canvas, interpolation='nearest')

# plot trails and agents as scatter
trailA_sc = ax.scatter([],[], s=20, marker='s')
trailB_sc = ax.scatter([],[], s=20, marker='s')
agent_sc = ax.scatter([],[], s=80, marker='o')

def update_frame(i):
    ax.clear()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Step {i+1}")
    # base
    display = canvas.copy()
    # draw keys from history[i] (uncollected set)
    if i < len(history):
        _, _, uncol = history[i]
    else:
        uncol = set()
    # draw keys: collected keys grey, uncollected red
    for (kr,kc) in keys:
        if (kr,kc) in uncol:
            display[kr,kc] = [1.0, 0.4, 0.4]  # red-ish
        else:
            display[kr,kc] = [0.8, 0.8, 0.8]  # grey when collected
    ax.imshow(display, interpolation='nearest')
    # trails
    ta = trailA[:i+1]
    tb = trailB[:i+1]
    if ta:
        ys = [p[0] for p in ta]; xs = [p[1] for p in ta]
        ax.plot(xs, ys, linestyle='-', linewidth=2, alpha=0.8, label='A trail', color='blue')
    if tb:
        ys = [p[0] for p in tb]; xs = [p[1] for p in tb]
        ax.plot(xs, ys, linestyle='-', linewidth=2, alpha=0.8, label='B trail', color='green')
    # agent markers
    if i < len(history):
        a_pos, b_pos, _ = history[i]
        ax.scatter([a_pos[1]], [a_pos[0]], s=120, marker='o', color='blue', label='A')
        ax.scatter([b_pos[1]], [b_pos[0]], s=120, marker='o', color='green', label='B')
    ax.legend(loc='upper right')

ani = animation.FuncAnimation(fig, update_frame, frames=max(1, len(history)), interval=120, repeat=False)
plt.show()

# Remove above two lines with this in collab
# from IPython.display import HTML
# HTML(ani.to_jshtml())


# Summary print
print("Simulation finished in steps:", step)
print("Agent A collected:", agents['A']['collected'])
print("Agent B collected:", agents['B']['collected'])
