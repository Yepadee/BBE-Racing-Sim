#
# Racing Simulator
#

# Import the Python OpenCL API
import pyopencl as cl

# Import the Python Maths Library (for vectors)
import numpy as np

# Import Standard Library to time the execution
from time import time

import json

#------------------------------------------------------------------------------

f = open('config.json', 'r', encoding='utf-8')
config = json.load(f)
f.close()

track_length = config["track_length"]
conditions = np.array(config["conditions"])
competetors = config["competetors"]
n_competetors = len(competetors)
preferences = np.array(competetors["preferences"])
distributions = competetors["distr"]

mag = np.sqrt(len(conditions))

def condition_score(x):
    return 1.0 - np.linalg.norm(conditions-x)/mag

print(preferences)
preference_scores = [condition_score(p) for p in preferences]
print("p: ", preference_scores)

NUM_RACES = 10

n_positions = NUM_RACES * n_competetors

print(n_competetors)

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
options = "-D n_c=%d -D l=%d -D offset=%d" % (n_competetors, track_length, 0)
program = cl.Program(context, kernelsource).build(options)

# Create initial positions vector to be returned from device
h_positions = np.zeros(n_positions).astype(np.float64)

# Create initial randoms vector to be returned from device
h_randoms = np.zeros(n_positions).astype(np.float64)

# Create the input (a, b) arrays in device memory and copy data from host
mf = cl.mem_flags
d_positions = cl.Buffer(context, mf.COPY_HOST_PTR, hostbuf=h_positions) # Read and write
d_randoms = cl.Buffer(context, mf.COPY_HOST_PTR, hostbuf=h_randoms) # Read and write

# Start the timer
rtime = time()

# Execute the kernel over the entire range of our 1d input
# allowing OpenCL runtime to select the work group items for the device
update_positions = program.update_positions
update_positions.set_scalar_arg_dtypes([None, None])

generate_randoms = program.generate_randoms
generate_randoms.set_scalar_arg_dtypes([None])

for i in range(9):
    generate_randoms(queue, h_randoms.shape, None, d_randoms)
    update_positions(queue, (NUM_RACES,), None, d_randoms, d_positions)

# Wait for the commands to finish before reading back
queue.finish()
rtime = time() - rtime
print("The kernel ran in", rtime, "seconds")

# Read back the results from the compute device
cl.enqueue_copy(queue, h_positions, d_positions)

# Find winners
def find_winners(positions):
    return [np.argmax(p) for p in positions]

winners = find_winners(h_positions)



# Test the results
print(h_positions)
print(np.mean(h_positions))
#print(winners)