import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os

def plot_odds(n_competetors: int, actions_per_period: int, odds: np.float32, fig_path: str) -> None:
    _, ax = plt.subplots()
    steps, _ = odds.shape
    xs: np.int32 = np.arange(0, steps*actions_per_period, actions_per_period)

    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1.*i/n_competetors) for i in range(n_competetors)])
    ax.step(xs, odds/100.0)
    ax.set_ylim([1,20])
    ax.legend(np.arange(1, n_competetors + 1), title='Horse', bbox_to_anchor=(1.00, 1), loc='upper left', fontsize='x-small')
    
    plt.yscale("log", base=2)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    
    ax.set_yticks([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])

    ax.set_ylabel('Decimal Odds')
    ax.set_xlabel('Time/s')
    
    dir_path = os.path.dirname(fig_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(fig_path)
    plt.close()

def plot_positions(n_competetors: int, actions_per_period: int, positions: np.float32, fig_path: str) -> None:
    _, ax = plt.subplots()
    steps, _ = positions.shape
    xs: np.int32 = np.arange(0, steps*actions_per_period, actions_per_period)

    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1.*i/n_competetors) for i in range(n_competetors)])
    ax.step(xs, positions)
    ax.legend(np.arange(1, n_competetors + 1), title='Horse', bbox_to_anchor=(1.00, 1), loc='upper left', fontsize='x-small')

    ax.set_ylabel('Position/m')
    ax.set_xlabel('Time/s')
    
    dir_path = os.path.dirname(fig_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(fig_path)
    plt.close()