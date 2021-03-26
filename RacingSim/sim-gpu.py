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
from sim_output import plot_winners

#------------------------------------------------------------------------------

# Open and parse config
f = open('config.json', 'r', encoding='utf-8')
config = json.load(f)
f.close()

def load_racesim(config):
    track_length = config["track_length"]
    track_width = config["track_width"]
    clean_air_dist = config["clean_air_dist"]

    n_steps = config["num_steps"]

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

    return track_params, competetor_params, n_steps

track_params, competetor_params, n_steps = load_racesim(config)

n_races = 10000
n_steps = 800

racesim = RaceSimParallel(n_steps, n_races, track_params, competetor_params)
competetor_positions = np.zeros(competetor_params.n_competetors).astype(np.float32)


winners = racesim.simulate_races(competetor_positions)

print(winners)
plot_winners(winners)
