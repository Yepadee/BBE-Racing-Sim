import numpy as np
import matplotlib.pyplot as plt
import os

def plot_winners(n_competetors, winners, fig_path):
    # Plot competetor win frequencies
    bins = np.arange(1, n_competetors + 1.5) - 0.5
    _, ax = plt.subplots()
    _ = ax.hist(winners, bins)
    ax.set_xticks(bins + 0.5)
    ax.set_ylabel('No. Wins')
    ax.set_xlabel('Horse No.')

    dir_path = os.path.dirname(fig_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(fig_path)
    plt.close()

def plot_winners_freqs(n_competetors, winner_freqs, fig_path):
    plt.bar(range(1, n_competetors + 1), winner_freqs)
    plt.ylabel('frequency')
    plt.xticks(np.arange(1, n_competetors + 2))

    dir_path = os.path.dirname(fig_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(fig_path)
    plt.close()
