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

f = open('config.json', 'r', encoding='utf-8')
config = json.load(f)
f.close()

n_races = config["num_races"]
n_steps = config["num_steps"]
track_length = config["track_length"]
track_width = config["track_width"]
clean_air_dist = config["clean_air_dist"]
conditions = np.array(config["conditions"])
competetors = config["competetors"]
n_competetors = competetors["quantity"]
preferences = np.array(competetors["preferences"])
dist_params = competetors["dist_params"]

mag = np.sqrt(len(conditions))

def condition_score(x):
    return 1.0 - np.linalg.norm(conditions-x)/mag

preference_scores = [condition_score(p) for p in preferences]
print("Preferences: ", preference_scores)

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

h_rng_mins = np.array([params[0] for params in dist_params]).astype(np.float32)
h_rng_maxs = np.array([params[1] for params in dist_params]).astype(np.float32)

# Create initial positions vector to be returned from device
h_positions = np.zeros(n_positions).reshape((n_races, n_competetors)).astype(np.float32)

h_winners = np.zeros(n_races).astype(np.int8)

# Create initial randoms vector to be returned from device
h_randoms = np.zeros(2 * n_positions).astype(np.float32)

# Create the input (a, b) arrays in device memory and copy data from host
mf = cl.mem_flags
d_winners = cl.Buffer(context, mf.COPY_HOST_PTR, hostbuf=h_winners) # Read and write
d_positions = cl.Buffer(context, mf.COPY_HOST_PTR, hostbuf=h_positions) # Read and write
d_tmp_positions = cl.Buffer(context, mf.COPY_HOST_PTR, hostbuf=h_positions) # Read and write
d_randoms = cl.Buffer(context, mf.COPY_HOST_PTR, hostbuf=h_randoms) # Read and write

d_rng_mins = cl.Buffer(context, mf.COPY_HOST_PTR, hostbuf=h_rng_mins) # Read and write
d_rng_maxs = cl.Buffer(context, mf.COPY_HOST_PTR, hostbuf=h_rng_maxs) # Read and write

# Start the timer
rtime = time()

offset = int(rtime)

# Execute the kernel over the entire range of our 1d input
# allowing OpenCL runtime to select the work group items for the device
generate_randoms = program.generate_randoms
generate_randoms.set_scalar_arg_dtypes([np.int64, None])

update_positions = program.update_positions
update_positions.set_scalar_arg_dtypes([None, None, None, None, None, None])

for i in range(n_steps // 2):
    offset += n_races
    generate_randoms(queue, h_randoms.shape, None,
        offset, d_randoms)
    update_positions(queue, (n_races,n_competetors), None,
        d_rng_mins, d_rng_maxs, d_randoms, d_positions, d_tmp_positions, d_winners)
    offset += n_races
    generate_randoms(queue, h_randoms.shape, None,
        offset, d_randoms)
    update_positions(queue, (n_races,n_competetors), None,
        d_rng_mins, d_rng_maxs, d_randoms, d_tmp_positions, d_positions, d_winners)

# Wait for the commands to finish before reading back
queue.finish()
rtime = time() - rtime
print("The kernel ran in", rtime, "seconds")

# Read back the results from the compute device
cl.enqueue_copy(queue, h_positions, d_positions)
cl.enqueue_copy(queue, h_winners, d_winners)

# Test the results
print(h_positions)
print(h_winners)
print("Avergage position: ", np.mean(h_positions))

not_complete = np.argwhere(h_winners == 0).flatten()
print("not finished: ", not_complete)

best_racer = stats.mode(h_winners)[0]
print("Best racer: ", best_racer)

#plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

bins = np.arange(1, 20 + 0.5) - 0.5

# then you plot away
fig, ax = plt.subplots()
_ = ax.hist(h_winners, bins)
ax.set_xticks(bins + 0.5)
plt.savefig('output/freq.png')

print("plt saved")