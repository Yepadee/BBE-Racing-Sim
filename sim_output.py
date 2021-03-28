import numpy as np
import matplotlib.pyplot as plt

def plot_winners(winners, fig_path):
    # Plot competetor win frequencies
    bins = np.arange(1, 20 + 0.5) - 0.5
    fig, ax = plt.subplots()
    _ = ax.hist(winners, bins)
    ax.set_xticks(bins + 0.5)
    plt.savefig(f'{fig_path}.png')
    plt.close()