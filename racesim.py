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
        self.__context = context
        self.__n_races = n_races
        self._track_params = track_params
        self._competetor_params = competetor_params
        self.__n_positions = n_races * competetor_params.n_competetors

        # Host Buffers
        self.__h_preferences = competetor_params.preference_scores
        self.__h_rngs = competetor_params.dist_params
        self.__h_resp_levels = competetor_params.resp_levels
        self.__h_resp_durations = competetor_params.resp_durations

        self._h_positions = None
        self._h_winners = None

        # Device Buffers
        mf = cl.mem_flags
        self.__d_preferences = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.__h_preferences) # Read Only
        self.__d_rngs = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.__h_rngs) # Read Only
        self.__d_resp_levels = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.__h_resp_levels) # Read Only
        self.__d_resp_durations = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.__h_resp_durations) # Read Only

        self._d_positions = None
        self.__d_tmp_positions = None
        self._d_winners = None

        self.set_competetor_positions(np.zeros(competetor_params.n_competetors).astype(np.float32))

        # Create a command queue
        self._queue = cl.CommandQueue(context)

        program = self.__build_program(context)
        self.update_positions = program.update_positions
        self.update_positions.set_scalar_arg_dtypes([None, None, None, None, None, None, None, np.int64])

        self.offset = int(time())

    def __build_program(self, context):
        # Load kernel
        f = open('kernel/kernels.cl', 'r', encoding='utf-8')
        kernelsource = ''.join(f.readlines())
        f.close()

        options = "-D n_c=%d -D n_r=%d -D l=%d -D w=%f -D clean_air_dist=%d" % (self._competetor_params.n_competetors, self.__n_races, self._track_params.length, self._track_params.width, self._track_params.clean_air_dist)
        return cl.Program(context, kernelsource).build(options)

    def __format_positions(self, positions):
        return np.tile(positions, (self.__n_races, 1)).astype(np.float32)

    def set_competetor_positions(self, competetor_positions):
        self._h_positions = self.__format_positions(competetor_positions)
        self._h_winners = np.zeros(self.__n_races).astype(np.int8)
        mf = cl.mem_flags
        self._d_positions = cl.Buffer(self.__context, mf.COPY_HOST_PTR, hostbuf=self._h_positions) # Read and write
        self.__d_tmp_positions = cl.Buffer(self.__context, mf.COPY_HOST_PTR, hostbuf=self._h_positions) # Read and write
        self._d_winners = cl.Buffer(self.__context, mf.COPY_HOST_PTR, hostbuf=self._h_winners) # Read and write

    def _step(self, n_steps):
        for i in range(n_steps):
            self.offset += 2*self.__n_positions
            self.update_positions(self._queue, (self.__n_races, self._competetor_params.n_competetors), None,
                self.__d_preferences, self.__d_rngs, self.__d_resp_levels, self.__d_resp_durations, self._d_positions, self.__d_tmp_positions, self._d_winners, self.offset)

            self.offset += 2*self.__n_positions
            self.update_positions(self._queue, (self.__n_races, self._competetor_params.n_competetors), None,
                self.__d_preferences, self.__d_rngs, self.__d_resp_levels, self.__d_resp_durations, self.__d_tmp_positions, self._d_positions, self._d_winners, self.offset)

    def _stop(self):
        self._queue.finish()


class RaceSimSerial(RaceSim):
    def __init__(self, track_params, competetor_params):
        context = cl.create_some_context() # TODO get cpu context
        super().__init__(context, 1, track_params, competetor_params)

    def get_competetor_positions(self):
        self._stop()
        cl.enqueue_copy(self._queue, self._h_positions, self._d_positions)
        return self._h_positions[0]

    def step(self, n_steps):
        self._step(n_steps)


class RaceSimParallel(RaceSim):
    def __init__(self, max_steps, n_races, track_params, competetor_params):
        self.max_steps = max_steps
        context = cl.create_some_context() # TODO get gpu context
        super().__init__(context, n_races, track_params, competetor_params)

    def __get_steps_remaining(self, max_steps, competetor_positions, track_length) -> int:
        min_pos = min(competetor_positions)
        percent_complete = min_pos / track_length
        return int(max_steps * (1.0 - percent_complete)) // 2

    def simulate_races(self, competetor_positions):
        self.set_competetor_positions(competetor_positions)

        rtime = time()

        n_steps = self.__get_steps_remaining(self.max_steps, competetor_positions, self._track_params.length)
        print("n_steps: ", n_steps)

        self._step(n_steps)
        self._stop()

        rtime = time() - rtime
        print("The kernel ran in", rtime, "seconds")

        return self.get_winners()

    def get_winners(self):
        cl.enqueue_copy(self._queue, self._h_winners, self._d_winners)
        return self._h_winners