import pyopencl as cl
import numpy as np
from time import time

class TrackParams(object):
    def __init__(self, length, width, clean_air_dist):
        self.length = length
        self.width = width
        self.clean_air_dist = clean_air_dist


class CompetetorParams(object):
    def __init__(self, n_competetors: int, track_conditions, track_preferences, dist_params, resp_levels, resp_durations):
        self.n_competetors = n_competetors
        mag = np.sqrt(len(track_conditions))

        def condition_score(x):
            return 1.0 - np.linalg.norm(track_conditions-x)/mag

        self.preference_scores = np.array([condition_score(p) for p in track_preferences]).astype(np.float32)
        self.dist_params = dist_params
        self.resp_levels = resp_levels
        self.resp_durations = resp_durations

class RaceSim(object):
    def __init__(self, context, n_races, track_params, competetor_params):
        self.context = context
        self.n_races = n_races
        self.track_params = track_params
        self.competetor_params = competetor_params
        self.n_positions = n_races * competetor_params.n_competetors

        # Host Buffers
        self.h_preferences = competetor_params.preference_scores
        self.h_rngs = competetor_params.dist_params
        self.h_resp_levels = competetor_params.resp_levels
        self.h_resp_durations = competetor_params.resp_durations

        self.h_positions = None
        self.h_winners = None

        # Device Buffers
        mf = cl.mem_flags
        self.d_preferences = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h_preferences) # Read Only
        self.d_rngs = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h_rngs) # Read Only
        self.d_resp_levels = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h_resp_levels) # Read Only
        self.d_resp_durations = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h_resp_durations) # Read Only

        self.d_positions = None
        self.d_tmp_positions = None
        self.d_winnder = None

        # Create a command queue
        self.queue = cl.CommandQueue(context)

        program = self.__build_program(context)
        self.update_positions = program.update_positions
        self.update_positions.set_scalar_arg_dtypes([None, None, None, None, None, None, None, np.int64])

        self.offset = int(time())

    def __build_program(self, context):
        # Load kernel
        f = open('RacingSim/kernel/kernels.cl', 'r', encoding='utf-8')
        kernelsource = ''.join(f.readlines())
        f.close()

        options = "-D n_c=%d -D n_r=%d -D l=%d -D w=%f -D clean_air_dist=%d" % (self.competetor_params.n_competetors, self.n_races, self.track_params.length, self.track_params.width, self.track_params.clean_air_dist)
        return cl.Program(context, kernelsource).build(options)

    def __format_positions(self, positions):
        return np.tile(positions, (self.n_races, 1)).astype(np.float32)

    def set_competetor_positions(self, competetor_positions):
        self.h_positions = self.__format_positions(competetor_positions)
        self.h_winners = np.zeros(self.n_races).astype(np.int8)
        mf = cl.mem_flags
        self.d_positions = cl.Buffer(self.context, mf.COPY_HOST_PTR, hostbuf=self.h_positions) # Read and write
        self.d_tmp_positions = cl.Buffer(self.context, mf.COPY_HOST_PTR, hostbuf=self.h_positions) # Read and write
        self.d_winners = cl.Buffer(self.context, mf.COPY_HOST_PTR, hostbuf=self.h_winners) # Read and write

    def step(self):
        self.offset += 2*self.n_positions

        self.update_positions(self.queue, (self.n_races, self.competetor_params.n_competetors), None,
            self.d_preferences, self.d_rngs, self.d_resp_levels, self.d_resp_durations, self.d_positions, self.d_tmp_positions, self.d_winners, self.offset)

        self.offset += 2*self.n_positions
        self.update_positions(self.queue, (self.n_races,self.competetor_params.n_competetors), None,
            self.d_preferences, self.d_rngs, self.d_resp_levels, self.d_resp_durations, self.d_tmp_positions, self.d_positions, self.d_winners, self.offset)

    def stop(self):
        self.queue.finish()

    def get_competetor_positions(self):
        cl.enqueue_copy(self.queue, self.h_positions, self.d_positions)
        return self.h_positions

    def get_winners(self):
        cl.enqueue_copy(self.queue, self.h_winners, self.d_winners)
        return self.h_winners

