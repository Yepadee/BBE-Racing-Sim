import pyopencl as cl
import numpy as np
from time import time
import json

def load_racesim():
    '''
    Open and parse racesim config
    Returns the racetrack and competetor parameters,
    also the number of steps required to run a full race
    '''
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

def get_gpu_context():
    '''Search for and return a gpu context'''
    all_platforms = cl.get_platforms()
    platform = next((p for p in all_platforms if
                     p.get_devices(device_type=cl.device_type.GPU) != []),
                     None)
    my_gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
    return cl.Context(devices=my_gpu_devices)

def get_cpu_context():
    '''Search for and return a cpu context'''
    all_platforms = cl.get_platforms()
    platform = next((p for p in all_platforms if
                     p.get_devices(device_type=cl.device_type.CPU) != []),
                     None)
    my_gpu_devices = platform.get_devices(device_type=cl.device_type.CPU)
    return cl.Context(devices=my_gpu_devices)

class TrackParams(object):
    def __init__(self, length: int, width: int, clean_air_dist: int):
        self.length = length
        self.width = width
        self.clean_air_dist = clean_air_dist

class CompetetorParams(object):
    def __init__(self, n_competetors: int, track_conditions: np.array(np.float32),
                 track_preferences: np.array(np.float32), dist_params: np.array(np.float32),
                 resp_levels: np.array(np.float32), resp_durations: np.array(np.float32)):
        self.n_competetors = n_competetors
        mag = np.sqrt(len(track_conditions))

        def condition_score(x):
            return 1.0 - np.linalg.norm(track_conditions-x)/mag

        self.preference_scores = np.array([condition_score(p) for p in track_preferences]).astype(np.float32)
        self.dist_params = dist_params
        self.resp_levels = resp_levels
        self.resp_durations = resp_durations

class RaceSim(object):
    def __init__(self, context: cl.Context, n_races: int, track_params: TrackParams, competetor_params: CompetetorParams):
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

    def __build_program(self, context: cl.Context) -> cl.Program:
        '''
        Load kernel and build program for chosen device (gpu/cpu)
        Returns resultant program
        '''
        # Load kernel
        f = open('kernel/kernels.cl', 'r', encoding='utf-8')
        kernelsource = ''.join(f.readlines())
        f.close()

        options = "-D n_c=%d -D n_r=%d -D l=%d -D w=%f -D clean_air_dist=%d" % (self._competetor_params.n_competetors, self.__n_races, self._track_params.length, self._track_params.width, self._track_params.clean_air_dist)
        return cl.Program(context, kernelsource).build(options)

    def __format_positions(self, positions: np.array(np.float32)) -> np.array(np.float32):
        '''
        Allocate memory for competetor positions using
        provided positions for each race instance
        '''
        return np.tile(positions, (self.__n_races, 1)).astype(np.float32)

    def set_competetor_positions(self, competetor_positions: np.array(np.float32)) -> None:
        self._h_positions = self.__format_positions(competetor_positions)
        self._h_winners = np.zeros(self.__n_races).astype(np.int8)
        mf = cl.mem_flags
        self._d_positions = cl.Buffer(self.__context, mf.COPY_HOST_PTR, hostbuf=self._h_positions) # Read and write
        self.__d_tmp_positions = cl.Buffer(self.__context, mf.COPY_HOST_PTR, hostbuf=self._h_positions) # Read and write
        self._d_winners = cl.Buffer(self.__context, mf.COPY_HOST_PTR, hostbuf=self._h_winners) # Read and write

    def _step(self, n_steps: int) -> None:
        '''Complete 'n_steps' of the simulation'''
        for i in range(n_steps):
            self.offset += 2*self.__n_positions
            self.update_positions(self._queue, (self.__n_races, self._competetor_params.n_competetors), None,
                self.__d_preferences, self.__d_rngs, self.__d_resp_levels, self.__d_resp_durations, self._d_positions, self.__d_tmp_positions, self._d_winners, self.offset)

            self.offset += 2*self.__n_positions
            self.update_positions(self._queue, (self.__n_races, self._competetor_params.n_competetors), None,
                self.__d_preferences, self.__d_rngs, self.__d_resp_levels, self.__d_resp_durations, self.__d_tmp_positions, self._d_positions, self._d_winners, self.offset)

    def _stop(self) -> None:
        '''Wait for the queue to finish so data may be transferred from the device'''
        self._queue.finish()


class RaceSimSerial(RaceSim):
    def __init__(self, track_params: TrackParams, competetor_params: CompetetorParams):
        context = get_cpu_context()
        super().__init__(context, 1, track_params, competetor_params)

    def get_competetor_positions(self) -> np.array(np.float32):
        '''
        Copy competetor positions from the device
        Returns the copied competetor positions
        '''
        self._stop()
        cl.enqueue_copy(self._queue, self._h_positions, self._d_positions)
        return self._h_positions[0]

    def step(self, n_steps) -> None:
        '''Complete 'n_steps' of the simulation'''
        self._step(n_steps)


class RaceSimParallel(RaceSim):
    def __init__(self, max_steps: int, n_races: int, track_params: TrackParams, competetor_params: CompetetorParams):
        self.max_steps = max_steps
        context = get_gpu_context()
        super().__init__(context, n_races, track_params, competetor_params)

    def __get_steps_remaining(self, max_steps: int, competetor_positions: np.array(np.float32), track_length: int) -> int:
        '''
        Calculate and return how many more steps of the simulation
        need to take place before all races have finished
        '''
        min_pos = min(competetor_positions)
        percent_complete = min_pos / track_length
        return int(max_steps * (1.0 - percent_complete)) // 2

    def simulate_races(self, competetor_positions: np.array(np.float32)) -> np.array(np.int8):
        '''
        Run the racing simulation with competetors starting
        from the positions defined in 'competetor_positions'
        '''
        self.set_competetor_positions(competetor_positions)

        rtime = time()

        n_steps = self.__get_steps_remaining(self.max_steps, competetor_positions, self._track_params.length)
        print("n_steps: ", n_steps)

        self._step(n_steps)
        self._stop()

        rtime = time() - rtime
        print("The kernel ran in", rtime, "seconds")

        return self.__get_winners()

    def __get_winners(self) -> np.array(np.int8):
        '''
        Copy the race winners from the device
        Returns the copied winners
        '''
        cl.enqueue_copy(self._queue, self._h_winners, self._d_winners)
        return self._h_winners

if __name__ == "__main__":
    from sim_output import plot_winners
    track_params, competetor_params, n_steps = load_racesim()

    n_races = 10000

    race_sim_serial = RaceSimSerial(track_params, competetor_params)
    race_sim_parallel = RaceSimParallel(n_steps, n_races, track_params, competetor_params)

    race_sim_serial.step(200)
    competetor_positions = race_sim_serial.get_competetor_positions()

    winners = race_sim_parallel.simulate_races(competetor_positions)

    print(winners)
    plot_winners(winners)
