import sys
sys.path.append('../BBE-Racing-Sim/')

from racesim import *
from output_odds import plot_odds, plot_positions

from functools import reduce


'''Load racesim config'''
track_params, competetor_params = load_racesim_params()
n_competetors: int = competetor_params.n_competetors

n_simulations = 1000

'''Create race simulation instances'''
race: RaceSimSerial = RaceSimSerial(track_params, competetor_params)
race_simulations: RaceSimParallel = RaceSimParallel(n_simulations, track_params, competetor_params)

opinion_update_period: int = 1

all_odds = []
all_positions = []
while not race.is_finished():
    '''Get the current competetor positions'''
    competetor_positions = race.get_competetor_positions()
    all_positions.append(competetor_positions)
    
    print("Running simulations...")
    '''Run all the simulations from these positions'''
    predicted_winners = race_simulations.simulate_races(competetor_positions)
    print("Simulations complete!")

    odds = calculate_decimal_odds(n_competetors, predicted_winners)
    all_odds.append(odds)

    race.step(opinion_update_period)

all_positions = np.array(all_positions)
all_odds = np.array(all_odds)

plot_odds(n_competetors, all_odds, "output/odds")
plot_positions(n_competetors, all_positions, "output/positions-odds")