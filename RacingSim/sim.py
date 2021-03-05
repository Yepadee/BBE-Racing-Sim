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

#------------------------------------------------------------------------------

# Open and parse config
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

dist_params = np.array(competetors["dist_params"])
rngs = dist_params.flatten()

responsiveness = competetors["responsiveness"]
resp_levels = np.array(responsiveness["levels"]).flatten()
resp_durations = np.array(responsiveness["durations"]).flatten()

mag = np.sqrt(len(conditions))

def condition_score(x):
    return 1.0 - np.linalg.norm(conditions-x)/mag

preference_scores = [condition_score(p) for p in preferences]

n_positions = n_races * n_competetors

#------------------------------------------------------------------------------
f = open('RacingSim/kernel/kernels.cl', 'r', encoding='utf-8')
kernelsource = ''.join(f.readlines())
f.close()

#------------------------------------------------------------------------------

# Main procedure

# Create a compute context
context = cl.create_some_context()

# Create a command queue
queue = cl.CommandQueue(context)

# Create the compute program from the source buffer
# and build it
options = "-D n_c=%d -D n_r=%d -D l=%d -D w=%f -D clean_air_dist=%d" % (n_competetors, n_races, track_length, track_width, clean_air_dist)
program = cl.Program(context, kernelsource).build(options)

# Host Buffers
h_preferences = np.array(preference_scores).astype(np.float32)
h_rngs = rngs.astype(np.float32)
h_resp_levels = resp_levels.astype(np.float32)
h_resp_durations = resp_durations.astype(np.float32)

# Device Buffers
mf = cl.mem_flags
d_preferences = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_preferences) # Read Only
d_rngs = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_rngs) # Read Only
d_resp_levels = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_resp_levels) # Read Only
d_resp_durations = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_resp_durations) # Read Only


update_positions = program.update_positions
update_positions.set_scalar_arg_dtypes([None, None, None, None, None, None, None, np.int64])

competetor_positions = np.zeros(n_competetors).astype(np.float32)

def format_positions(positions):
    return np.tile(positions, (n_races, 1)).astype(np.float32)

def simulate_race(h_positions):
    h_winners = np.zeros(n_races).astype(np.int8)
    d_positions = cl.Buffer(context, mf.COPY_HOST_PTR, hostbuf=h_positions) # Read and write
    d_tmp_positions = cl.Buffer(context, mf.COPY_HOST_PTR, hostbuf=h_positions) # Read and write
    d_winners = cl.Buffer(context, mf.COPY_HOST_PTR, hostbuf=h_winners) # Read and write

    # Start the timer
    rtime = time()

    offset = int(rtime)

    for i in range(n_steps // 2):
        offset += 2*n_positions
        update_positions(queue, (n_races,n_competetors), None,
            d_preferences, d_rngs, d_resp_levels, d_resp_durations, d_positions, d_tmp_positions, d_winners, offset)

        offset += 2*n_positions
        update_positions(queue, (n_races,n_competetors), None,
            d_preferences, d_rngs, d_resp_levels, d_resp_durations, d_tmp_positions, d_positions, d_winners, offset)

    # Wait for the commands to finish before reading back
    queue.finish()
    rtime = time() - rtime
    print("The kernel ran in", rtime, "seconds")

    # Read back the results from the compute device
    cl.enqueue_copy(queue, h_positions, d_positions)
    cl.enqueue_copy(queue, h_winners, d_winners)

    # Print results
    #print(h_positions)
    
    print("Avergage position: ", np.mean(h_positions))

    not_complete = np.argwhere(h_winners == 0).flatten()
    print("not finished: ", not_complete)

    best_racer = stats.mode(h_winners)[0]
    print("Best racer: ", best_racer)

    return h_winners

formatted_positions = format_positions(competetor_positions)
winners = simulate_race(formatted_positions)

# Plot competetor win frequencies
bins = np.arange(1, 20 + 0.5) - 0.5
fig, ax = plt.subplots()
_ = ax.hist(winners, bins)
ax.set_xticks(bins + 0.5)
plt.savefig('output/resp/freq.png')

