# cooperative_path_planning.py
import heapq
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# %matplotlib inline
# (Add this line at the top of collab google)

# ---------------- CONFIG ----------------
R, C = 18, 26
WALL_PROB = 0.08
SEED = 3
random.seed(SEED); np.random.seed(SEED)

dirs = [(1,0),(-1,0),(0,1),(0,-1)]
def inb(r,c): return 0<=r<R and 0<=c<C

# Manhattan heuristic
def h(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

# ---------------- A* PATHFINDER ----------------
def astar(grid, start, goal):
    pq = [(h(start,goal), 0, start)]
    came = {start:None}
    g = {start:0}
    while pq:
        _, cost, cur = heapq.heappop(pq)
        if cur == goal: break
        for dr,dc in dirs:
            nb = (cur[0]+dr, cur[1]+dc)
            if not inb(*nb) or grid[nb]==1: continue
            ng = cost + 1
            if nb not in g or ng < g[nb]:
                g[nb] = ng
                came[nb] = cur
                heapq.heappush(pq, (ng + h(nb,goal), ng, nb))
    if goal not in came: return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came[cur]
    return path[::-1]

# ---------------- GRID SETUP ----------------
grid = np.zeros((R,C), dtype=int)
for r in range(R):
    for c in range(C):
        if random.random() < WALL_PROB:
            grid[r,c] = 1

startA, goalA = (0,0), (R-2, C-2)
startB, goalB = (R-1,0), (0, C-3)
grid[startA] = grid[startB] = grid[goalA] = grid[goalB] = 0

pathA = astar(grid, startA, goalA)
pathB = astar(grid, startB, goalB)

if pathA is None or pathB is None:
    print("One path is blocked. Try lowering WALL_PROB.")
    exit()

# ---------------- AGENTS ----------------
agents = {
    'A': {'pos': startA, 'path': pathA, 'idx': 0},
    'B': {'pos': startB, 'path': pathB, 'idx': 0},
}

history = []

# ---------------- SIMULATION LOOP ----------------
doneA = doneB = False
MAX_STEPS = 600
step = 0

while step < MAX_STEPS and (not doneA or not doneB):
    step += 1

    # Propose moves
    prop = {}
    for aid in ['A','B']:
        ag = agents[aid]
        if ag['idx'] < len(ag['path']) - 1:
            prop[aid] = ag['path'][ag['idx'] + 1]
        else:
            prop[aid] = ag['pos']  # already at goal

    # Collision avoidance
    # Case 1: both want the same next cell
    if prop['A'] == prop['B'] and prop['A'] != agents['A']['pos']:
        # Priority: the one with shorter remaining path moves
        remA = len(pathA) - agents['A']['idx']
        remB = len(pathB) - agents['B']['idx']
        if remA <= remB:
            # A moves, B waits
            prop['B'] = agents['B']['pos']
        else:
            prop['A'] = agents['A']['pos']

    # Case 2: swap conflict (A wants B's position and B wants A's position)
    if prop['A'] == agents['B']['pos'] and prop['B'] == agents['A']['pos']:
        # Simple rule: A waits
        prop['A'] = agents['A']['pos']

    # Apply moves
    for aid in ['A','B']:
        ag = agents[aid]
        if prop[aid] != ag['pos']:
            ag['idx'] += 1
            ag['pos'] = prop[aid]

    # Check goals
    if agents['A']['pos'] == goalA: doneA = True
    if agents['B']['pos'] == goalB: doneB = True

    history.append((agents['A']['pos'], agents['B']['pos']))

# ---------------- PRINT RESULTS ----------------
print("Steps taken:", step)
print("Agent A reached goal:", doneA, " at ", agents['A']['pos'])
print("Agent B reached goal:", doneB, " at ", agents['B']['pos'])

# ---------------- VISUALIZATION ----------------
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xticks([]); ax.set_yticks([])

base = np.ones((R,C,3))
for r in range(R):
    for c in range(C):
        if grid[r,c]==1: base[r,c] = [0,0,0]

def update(i):
    ax.clear()
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(base, interpolation='nearest')

    a_pos, b_pos = history[i]
    # draw paths
    ax.scatter(goalA[1], goalA[0], s=100, marker='*', color='blue')
    ax.scatter(goalB[1], goalB[0], s=100, marker='*', color='green')

    # plot trail
    trailA = [pos for pos,_ in history[:i+1]]
    trailB = [pos for _,pos in history[:i+1]]
    ax.plot([p[1] for p in trailA], [p[0] for p in trailA], color='blue', linewidth=2)
    ax.plot([p[1] for p in trailB], [p[0] for p in trailB], color='green', linewidth=2)

    ax.scatter(a_pos[1], a_pos[0], s=160, color='blue', marker='o', label='A')
    ax.scatter(b_pos[1], b_pos[0], s=160, color='green', marker='o', label='B')

    ax.set_title(f"Step {i+1}")
    ax.legend(loc='lower right')

ani = animation.FuncAnimation(fig, update, frames=len(history), interval=150, repeat=False)
plt.show()

# (Replace above 2 lines in Google with this)
# ani = animation.FuncAnimation(fig, update, frames=len(history), interval=150, repeat=False)
# from IPython.display import HTML
# HTML(ani.to_jshtml())
