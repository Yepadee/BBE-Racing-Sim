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

from racesim import RaceSimSerial, RaceSimParallel, TrackParams, CompetetorParams
from sim_output import plot_winners

#------------------------------------------------------------------------------


def load_racesim():
    # Open and parse config
    f = open('resources/config.json', 'r', encoding='utf-8')
    config = json.load(f)
    f.close()

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


if __name__ == "__main__":
    track_params, competetor_params, n_steps = load_racesim()

    n_races = 10000

    race_sim_serial = RaceSimSerial(track_params, competetor_params)
    race_sim_parallel = RaceSimParallel(n_steps, n_races, track_params, competetor_params)

    race_sim_serial.step(200)
    competetor_positions = race_sim_serial.get_competetor_positions()

    winners = race_sim_parallel.simulate_races(competetor_positions)

    print(winners)
    plot_winners(winners)
