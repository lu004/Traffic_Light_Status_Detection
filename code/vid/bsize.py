import numpy as np
bsize = np.loadtxt("/sig_eval/bsize.txt")

w = range(0, int(bsize.max()) + 10, 5)
print("w", "num")
for i in range(len(w)-1):
    d = bsize[np.logical_and(bsize[:] >= w[i], bsize[:] < w[i+1])]
    print(str(w[i]) + "-" + str(w[i+1]-1), len(d))
