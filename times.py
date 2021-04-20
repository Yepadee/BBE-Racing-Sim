import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter
import os


# times = np.array([1.020, 1.069, 1.301, 1.456, 1.514, 1.847, 2.495, 2.695])
# times1 = np.array([1.025, 1.0623, 1.290, 1.463, 1.488, 1.848, 2.431, 2.704])
# times2 = np.array([1.032, 1.085, 1.300, 1.444, 1.479, 1.842, 2.439, 2.724])

times = np.array([9.002, 9.359, 9.684, 10.401, 12.355, 15.879, 20.963, 24.035])
times1 = np.array([9.029, 9.073, 9.662, 10.355, 12.315, 15.866, 20.976, 23.759])
times2 = np.array([9.175, 9.137, 9.739, 10.255, 12.411, 15.827, 21.026, 23.800])

all_times = np.array([times, times1, times2])

avg_times = np.average(all_times, axis=0)

_, ax = plt.subplots()
xs = np.arange(5, 45, 5)

ax.scatter(xs, avg_times)
ax.set_ylabel('Time/s')
ax.set_xlabel('No. Horses')

plt.yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_formatter(ScalarFormatter())


plt.savefig("output/times.png")
plt.close()