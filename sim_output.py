import numpy as np
import matplotlib.pyplot as plt

def plot_winners(winners):
    # Plot competetor win frequencies
    bins = np.arange(1, 20 + 0.5) - 0.5
    fig, ax = plt.subplots()
    _ = ax.hist(winners, bins)
    ax.set_xticks(bins + 0.5)
    plt.savefig('output/resp2/freq.png')