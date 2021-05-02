import pyopencl as cl
import numpy as np
from time import time
import random
import json
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

def pick_winner(potential_winners_bits):
    '''
    More than one competetor may have crossed the finish line
    in the last simulated moment.
    From these winners, pick one at random.
    '''

    potential_winners = []
    for i, c in enumerate(bin(potential_winners_bits)[:1:-1], 1):
        if c == '1':
            potential_winners.append(i)
    return random.choice(potential_winners)

def prob_to_odds(prob: np.float32) -> np.int32:
    '''
    Calculate and return decimal odds from event probability.
    Cap an event with 0 prob to odds of 1000.
    '''
    if prob == 0:
        prob = 0.0001
    inverse = 1.0 / prob
    if inverse < 1000:
        integer_odds = round(inverse * 100)
        if integer_odds == 100:
            integer_odds += 1
        return integer_odds
    else:
        return 1000 * 100

def calculate_decimal_odds(n_events: int, predicted_winners: np.int8) -> np.int32:
    winner_freqs: np.int32 = np.zeros(n_events)
    for i in range(n_events):
        event_id = i + 1
        winner_freqs[i] = float(np.count_nonzero(predicted_winners == event_id))
    probs: np.int32 =  winner_freqs / predicted_winners.size
    return np.array([prob_to_odds(prob) for prob in probs])

def load_racesim_params():
    '''
    Open and parse racesim config
    Returns the racetrack and competetor parameters,
    also the number of steps required to run a full race
    '''
    # path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # print(path)
    f = open('resources/racesim-config.json', 'r', encoding='utf-8')
    config = json.load(f)
    f.close()

    track_length = config["track_length"]
    track_width = config["track_width"]
    clean_air_dist = config["clean_air_dist"]

    n_steps = config["num_steps"]

    competetors = config["competetors"]
    n_competetors = competetors["quantity"]

    preference_weight = config["preference_weight"]
    max_condition_value = config["max_condition_value"]
    conditions = np.array(config["conditions"]).astype(np.float32)
    preferences = np.array(competetors["preferences"][:n_competetors]).astype(np.float32)

    dist_params = np.array(competetors["dist_params"][:n_competetors]).flatten().astype(np.float32) / 2.0

    responsiveness = competetors["responsiveness"]
    resp_levels = np.array(responsiveness["levels"][:n_competetors]).flatten().astype(np.float32)
    resp_durations = np.array(responsiveness["durations"][:n_competetors]).flatten().astype(np.float32)

    track_params = TrackParams(track_length, track_width, clean_air_dist, n_steps)
    competetor_params = CompetetorParams(n_competetors, preference_weight, max_condition_value, conditions, preferences, dist_params, resp_levels, resp_durations)

    return track_params, competetor_params

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
    my_cpu_devices = platform.get_devices(device_type=cl.device_type.CPU)
    print(my_cpu_devices)
    return cl.Context(devices=my_cpu_devices)

class TrackParams(object):
    def __init__(self, length: int, width: int, clean_air_dist: int, n_steps: int):
        self.length = length
        self.width = width
        self.clean_air_dist = clean_air_dist
        self.n_steps = n_steps

class CompetetorParams(object):
    def __init__(self, n_competetors: int, preference_weight: float, max_condition_value: float,
            track_conditions: np.float32, track_preferences: np.float32,
            dist_params: np.float32, resp_levels: np.float32,
            resp_durations: np.float32
        ):
        
        self.n_competetors = n_competetors

        def condition_score(x):
            score = 1.0 - np.linalg.norm(abs(track_conditions-x))/max_condition_value
            return (1.0 - preference_weight) + preference_weight * score

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
        kernel_locaiton = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        kernel_locaiton += "/kernel"
        f = open(f'{kernel_locaiton}/kernels.cl', 'r', encoding='utf-8')
        kernelsource = ''.join(f.readlines())
        kernelsource = kernelsource.replace("<kernel_location>", kernel_locaiton)
        f.close()

        options = "-D n_c=%d -D n_r=%d -D l=%d -D w=%f -D clean_air_dist=%d" % (self._competetor_params.n_competetors, self.__n_races, self._track_params.length, self._track_params.width, self._track_params.clean_air_dist)
        return cl.Program(context, kernelsource).build(options)

    def __format_positions(self, positions: np.float32) -> np.float32:
        '''
        Allocate memory for competetor positions using
        provided positions for each race instance
        '''
        return np.tile(positions, (self.__n_races, 1)).astype(np.float32)

    def set_competetor_positions(self, competetor_positions: np.float32) -> None:
        self._h_positions = self.__format_positions(competetor_positions)
        self._h_winners = np.zeros(self.__n_races).astype(np.int64)
        mf = cl.mem_flags
        self._d_positions = cl.Buffer(self.__context, mf.COPY_HOST_PTR, hostbuf=self._h_positions) # Read and write
        self.__d_tmp_positions = cl.Buffer(self.__context, mf.COPY_HOST_PTR, hostbuf=self._h_positions) # Read and write
        self._d_winners = cl.Buffer(self.__context, mf.COPY_HOST_PTR, hostbuf=self._h_winners) # Read and write

    def _step(self, n_steps: int) -> None:
        '''Complete 'n_steps' of the simulation'''
        for i in range(n_steps):
            self.offset += 2*self.__n_positions
            self.update_positions(self._queue, (self.__n_races,), None,
                self.__d_preferences, self.__d_rngs, self.__d_resp_levels, self.__d_resp_durations, self._d_positions, self.__d_tmp_positions, self._d_winners, self.offset)

            self.offset += 2*self.__n_positions
            self.update_positions(self._queue, (self.__n_races,), None,
                self.__d_preferences, self.__d_rngs, self.__d_resp_levels, self.__d_resp_durations, self.__d_tmp_positions, self._d_positions, self._d_winners, self.offset)

    def _stop(self) -> None:
        '''Wait for the queue to finish so data may be transferred from the device'''
        self._queue.finish()

    def _get_winners(self) -> np.array(np.int8):
        '''
        Copy the race winners from the device
        Returns the copied winners
        '''
        cl.enqueue_copy(self._queue, self._h_winners, self._d_winners)
        return self._h_winners

class RaceSimSerial(RaceSim):
    def __init__(self, track_params: TrackParams, competetor_params: CompetetorParams):
        context = get_cpu_context()
        super().__init__(context, 1, track_params, competetor_params)

    def __load_competetor_positions(self) -> None:
        '''Copy competetor positions from the device'''
        self._stop()
        cl.enqueue_copy(self._queue, self._h_positions, self._d_positions)

    def get_competetor_positions(self) -> np.array(np.float32):
        '''Returns the competetor positions'''
        return np.copy(self._h_positions[0])

    def get_percent_complete(self) -> float:
        positions: np.float32 = self.get_competetor_positions()
        max_position = np.max(positions)
        length = self._track_params.length
        max_position = max_position if max_position < length else length
        return max_position / length

    def step(self, n_steps) -> None:
        '''Complete 'n_steps' of the simulation'''
        self._step(n_steps)
        self.__load_competetor_positions()

    def is_finished(self) -> bool:
        return self.get_winner() != 0

    def get_winner(self) -> int:
        '''
        Return the race winner from the device
        '''
        winners = self._get_winners()[0]
        if winners != 0:
            return pick_winner(winners)
        else:
            return winners

class RaceSimParallel(RaceSim):
    def __init__(self, n_races: int, track_params: TrackParams, competetor_params: CompetetorParams):
        context = get_gpu_context()
        super().__init__(context, n_races, track_params, competetor_params)

    def __get_steps_remaining(self, max_steps: int, competetor_positions: np.float32, track_length: int) -> int:
        '''
        Calculate and return how many more steps of the simulation
        need to take place before all races have finished
        '''
        min_pos = min(competetor_positions)
        percent_complete = min_pos / track_length
        return int(max_steps * (1.0 - percent_complete))

    def simulate_races(self, competetor_positions: np.float32) -> np.array(np.int8):
        '''
        Run the racing simulation with competetors starting
        from the positions defined in 'competetor_positions'
        '''
        self.set_competetor_positions(competetor_positions)

        rtime = time()

        n_steps = self.__get_steps_remaining(self._track_params.n_steps, competetor_positions, self._track_params.length)
        print("n_steps: ", n_steps)

        self._step(n_steps)
        self._stop()

        rtime = time() - rtime
        print("The kernel ran in", rtime, "seconds")

        winners =  self._get_winners()
        num_incomplete: int = np.sum(winners == 0)
        if np.count_nonzero(self._h_winners == 0):
            raise Exception(f"{num_incomplete} races did not finish. Consider increaseing 'num_steps'")

        return np.array([pick_winner(winner) for winner in self._h_winners]).astype(np.int8)

if __name__ == "__main__":
    from sim_output import plot_winners
    track_params, competetor_params = load_racesim_params()

    n_races = 100000

    race_sim_serial = RaceSimSerial(track_params, competetor_params)
    race_sim_parallel = RaceSimParallel(n_races, track_params, competetor_params)

    GPU = True

    if GPU:
        competetor_positions = race_sim_serial.get_competetor_positions()
        winners = race_sim_parallel.simulate_races(np.zeros(competetor_params.n_competetors))
        plot_winners(competetor_params.n_competetors, winners, "output/fig")
    else:
        rtime = time()
        for i in range(n_races):
            race_sim_serial.step(track_params.n_steps)
        rtime = time() - rtime
        print("The kernel ran in", rtime, "seconds")
