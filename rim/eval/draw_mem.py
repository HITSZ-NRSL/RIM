# read output/mem_usage.txt and draw the memory usage curve
# Usage: python3 draw_mem.py

import matplotlib.pyplot as plt

linestyle = "-"
linewidth = 1
marker = "o"
markersize = 1

# read output/mem_usage.txt and
x = []
cpu_mem = []
gpu_mem = []
with open("output/mem_usage.txt", "r") as f:
    f.readline()
    for line in f:
        line = line.strip()
        if line:
            x.append(int(line.split()[0]))
            cpu_mem.append(float(line.split()[1]))
            gpu_mem.append(float(line.split()[2]))

plt.plot(x, cpu_mem, "-.", label="System Memory Usage", color="red")

plt.plot(x, gpu_mem, "-.", label="Video Memory Usage", color="blue")
plt.xlabel("Frame Index", fontdict={"size": 13})
plt.ylabel("Memory Usage (GB)", fontdict={"size": 13})
plt.ylim(ymin=0, ymax=12)
plt.xlim(xmin=0, xmax=4540)


plt.legend(prop={"size": 11})
plt.show()
plt.savefig("output/mem_usage.png", dpi=300)
