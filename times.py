import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# times = np.array([1.020, 1.069, 1.301, 1.456, 1.514, 1.847, 2.495, 2.695])
# times1 = np.array([1.025, 1.0623, 1.290, 1.463, 1.488, 1.848, 2.431, 2.704])
# times2 = np.array([1.032, 1.085, 1.300, 1.444, 1.479, 1.842, 2.439, 2.724])

# times = np.array([9.002, 9.359, 9.684, 10.401, 12.355, 15.879, 20.963, 24.035])
# times1 = np.array([9.029, 9.073, 9.662, 10.355, 12.315, 15.866, 20.976, 23.759])
# times2 = np.array([9.175, 9.137, 9.739, 10.255, 12.411, 15.827, 21.026, 23.800])

gpu_times = np.array([1.05, 1.11, 1.35, 1.51, 1.54, 1.94, 2.61, 2.85]) / 10000
cpu_p_times = np.array([6.75, 6.87, 7.33, 7.09, 7.75, 7.86, 8.17, 8.21]) / 10000
cpu_times = np.array([111.8,114.2,113.1,112.9,113.9, 114.2, 112.4,113.7]) / 10000
gpu_s_times = np.array([22.2, 22.4, 22.7, 22.5, 22.8, 22.6, 22.5, 23.3]) / 100

times = np.array([9.499, 9.734, 9.964, 10.567, 12.808, 16.806, 22.089, 25.048])
times1 = np.array([9.296, 9.566, 10.184, 10.592, 12.871, 16.677, 22.136, 24.882])
times2 = np.array([9.544, 9.616, 10.227, 10.644, 12.823, 16.796, 22.120, 24.943])

soa_times = np.array([9.339, 9.478, 9.703, 9.835, 10.230, 10.641, 11.271, 11.639])
soa_times1 = np.array([9.340, 9.481, 9.700, 9.690, 10.084, 10.469, 11.116, 11.703])
soa_times2 = np.array([9.326, 9.478, 9.699, 9.778, 10.158, 10.687, 11.091, 11.751])

all_times = np.array([times, times1, times2])

soa_all_times = np.array([soa_times, soa_times1, soa_times2])

avg_times = np.average(all_times, axis=0)
soa_avg_times = np.average(soa_all_times, axis=0)

_, ax = plt.subplots()
xs = np.arange(5, 45, 5)

#ax.scatter(xs, avg_times)
ax.scatter(xs, soa_avg_times)
# ax.scatter(xs, cpu_times)
# ax.scatter(xs, cpu_p_times)



ax.set_ylabel('Time/s')
ax.set_xlabel('No. Horses')

# ax.legend(["GPU Serial Times",
#             "CPU Serial Times",
#             "CPU Parallel Times",
#             "GPU Parallel Times"], bbox_to_anchor=(0, 0.9), loc='upper left', fontsize='small')

# ax.legend(["GPU AOS",
#            "GPU SOA"], bbox_to_anchor=(0, 0.9), loc='upper left', fontsize='small')

# plt.yscale("log")
# ax.yaxis.set_major_formatter(ScalarFormatter())
# ax.yaxis.set_minor_formatter(ScalarFormatter())
#ax.set_yticks([0.001, 0.01, 0.1, 1.0, 10.0])

plt.savefig("output/times-soa-only.png")
plt.close()