#
# Racing Simulator
#

# Import the Python OpenCL API
import pyopencl as cl

# Import the Python Maths Library (for vectors)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Import Standard Library to time the execution
from time import time

import json

from racesim import *

#------------------------------------------------------------------------------

# Open and parse config
f = open('config.json', 'r', encoding='utf-8')
config = json.load(f)
f.close()

def parse_config(config):
    track_length = config["track_length"]
    track_width = config["track_width"]
    clean_air_dist = config["clean_air_dist"]

    competetors = config["competetors"]
    n_competetors = competetors["quantity"]

    conditions = np.array(config["conditions"]).astype(np.float32)
    preferences = np.array(competetors["preferences"]).astype(np.float32)

    dist_params = np.array(competetors["dist_params"]).flatten().astype(np.float32)

    responsiveness = competetors["responsiveness"]
    resp_levels = np.array(responsiveness["levels"]).flatten().astype(np.float32)
    resp_durations = np.array(responsiveness["durations"]).flatten().astype(np.float32)

    track_params = TrackParams(track_length, track_width, clean_air_dist)
    competetor_params = CompetetorParams(n_competetors, conditions, preferences, dist_params, resp_levels, resp_durations)

    return track_params, competetor_params

track_params, competetor_params = parse_config(config)

n_races = 10000
n_competetors = competetor_params.n_competetors

n_steps = 800

n_positions = n_races * n_competetors

# Main procedure

# Create a compute context
context = cl.create_some_context()

competetor_positions = np.zeros(n_competetors).astype(np.float32)

racesim = RaceSim(context, 10000, track_params, competetor_params)
racesim.set_competetor_positions(competetor_positions)

n_steps = 800

for i in range(n_steps//2):
    racesim.step()

racesim.stop()

print(racesim.get_winners())

# Plot competetor win frequencies
# bins = np.arange(1, 20 + 0.5) - 0.5
# fig, ax = plt.subplots()
# _ = ax.hist(winners, bins)
# ax.set_xticks(bins + 0.5)
# plt.savefig('output/resp/freq.png')

