import matplotlib.pyplot as plt

# Workload sizes (x-axis)
x_ticks = [2,4,6,8,10,12,14,16]
x = [0.4, 0.8, 1.2, 1.6, 2, 5, 7.5, 15]  # numeric values for plotting

# Data from the image
libcudf_like = [11.1, 17.4, 24.85, 30.75, 36.35, 87.1, 333.6, 651.9]
libcudf_contains = [3.1, 5.5, 7.65, 8.75, 10.8, 26.05, 34.75, 68.95]
devesh_like = [7.2, 12.7, 18.55, 23.1, 28, 69.05, 87.35, 173.4]
devesh_contains = [4.9, 9, 13.15, 16.2, 19.25, 42.8, 56.2, 114.15]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, libcudf_like, label="libcudf like", marker='o',color="yellowgreen")
plt.plot(x, libcudf_contains, label="libcudf contains", marker='o',color="mediumpurple")
plt.plot(x, devesh_like, label="devesh like", marker='o',color="darkolivegreen")
plt.plot(x, devesh_contains, label="devesh contains", marker='o',color="rebeccapurple")

plt.xticks(x_ticks, x_ticks)
plt.xlabel("Workload Size")
plt.ylabel("Time (ms)")
plt.title("Performance Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("perf_graph_07-31.pdf")