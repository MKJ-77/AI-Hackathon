import numpy as np
import pandas as pd
from tabulate import tabulate

# Grid and agent/item definitions
GRID_SIZE = (10, 10)
agents = [(0, 0), (9, 0)]  # 2 agents
items = [((2, 2), (7, 7)), ((8, 1), (1, 4)), ((5, 8), (9, 9)),
         ((3, 6), (2, 8)), ((1, 8), (4, 1))]  # 5 tasks

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Proximity-based task assignment (nearest agent for each pickup)
agent_tasks = {i: [] for i in range(len(agents))}
agent_positions = list(agents)
for idx, (pick, drop) in enumerate(items):
    distances = [manhattan_distance(agent_positions[a], pick) for a in range(len(agents))]
    best_agent = np.argmin(distances)
    agent_tasks[best_agent].append(idx)
    agent_positions[best_agent] = drop

# Simulation: move agents, track total travel and time
results = []
for agent_id, tasks in agent_tasks.items():
    pos = agents[agent_id]
    total_dist = 0
    total_time = 0
    for idx in tasks:
        pick, drop = items[idx]
        d1 = manhattan_distance(pos, pick)
        d2 = manhattan_distance(pick, drop)
        total_dist += d1 + d2
        total_time += d1 + d2
        pos = drop
    efficiency = len(tasks) / (total_time / 60) if total_time else 0  # Units/hour
    results.append([
        str(agent_id + 1),
        str(len(tasks)),
        str(total_dist),
        str(total_time),
        "{:.2f}".format(efficiency)
    ])

# Fill table with total row
total_tasks = sum(int(row[1]) for row in results)
total_dist = sum(int(row[2]) for row in results)
total_time = sum(int(row[3]) for row in results)
total_eff = round(total_tasks / (total_time / 60), 2) if total_time > 0 else 0
results.append(["-", str(total_tasks), str(total_dist), str(total_time), str(total_eff)])

headers = ["Agent", "Tasks Picked", "Total Distance", "Total Time", "Efficiency (Units/Hr)"]
print(tabulate(results, headers=headers, tablefmt="grid"))