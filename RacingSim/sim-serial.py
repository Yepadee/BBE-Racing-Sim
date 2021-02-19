# Import the Python Maths Library (for vectors)
import numpy as np
import matplotlib.pyplot as plt

# Import Standard Library to time the execution
from time import time

from random import random

import json

f = open('config.json', 'r', encoding='utf-8')
config = json.load(f)
f.close()

n_races = config["num_races"]
n_steps = config["num_steps"]

track_length = config["track_length"]
track_width = config["track_width"]
clean_air_dist = config["clean_air_dist"]

competetors = config["competetors"]
n_competetors = competetors["quantity"]

conditions = np.array(config["conditions"])
preferences = np.array(competetors["preferences"])

dist_params = competetors["dist_params"]
rng_mins = [params[0] for params in dist_params]
rng_maxs = [params[1] for params in dist_params]

mag = np.sqrt(len(conditions))

def condition_score(x):
    return 1.0 - np.linalg.norm(conditions-x)/mag

preference_scores = [condition_score(p) for p in preferences]

n_positions = n_races * n_competetors

positions = np.zeros(n_positions).astype(np.float32)
tmp_positions = np.zeros(n_positions).astype(np.float32)

def u(min, max):
    rnd = random()
    return min + (rnd * (max - min))

def g(positions, r, c):
    pos = positions[c + r*n_competetors]
    total_distances = 0.0
    num_close_infront = 0
    for c2 in range(n_competetors):
        distance_from_c2 = positions[c2 + r*n_competetors] - pos
        if (distance_from_c2 > 0 and distance_from_c2 < clean_air_dist):
            num_close_infront += 1
            total_distances += distance_from_c2
    
    if num_close_infront==0:
        avg_distance = 0
    else:
        avg_distance = total_distances / num_close_infront
    blockage_factor = (clean_air_dist - avg_distance) / clean_air_dist
    prob = num_close_infront / track_width
    rdm = random()

    if rdm < prob:
        return blockage_factor
    else:
        return 1.0

def update_competetor(c, r, rng_mins, rng_maxs, positions, tmp_positions, winners):
    winner = winners[r]
    no_winner_mask = winner == 0
    i = c + r*n_competetors
    tmp_positions[i] = positions[i] + no_winner_mask * preference_scores[c] * u(rng_mins[c], rng_maxs[c]) * g(positions, r, c)
    if (tmp_positions[i] >= track_length):
         winners[r] = (c + 1)

# Start the timer
rtime = time()
winners = np.zeros(n_races)

for t in range(n_steps // 2):
    for r in range(n_races):
        for c in range(n_competetors):
            update_competetor(c, r, rng_mins, rng_maxs, positions, tmp_positions, winners)
            update_competetor(c, r, rng_mins, rng_maxs, tmp_positions, positions, winners)
            

rtime = time() - rtime
print("The kernel ran in", rtime, "seconds")

# Plot competetor win frequencies
bins = np.arange(1, 20 + 0.5) - 0.5
fig, ax = plt.subplots()
_ = ax.hist(winners, bins)
ax.set_xticks(bins + 0.5)
plt.savefig('output/freq-serial.png')