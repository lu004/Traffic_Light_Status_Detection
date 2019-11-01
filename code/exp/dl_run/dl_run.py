import os
import numpy as np

data = "/sig/exp/dl_run/"

with open(os.path.join(data, "fps.txt")) as f:
    l = f.readlines()
l = np.array(l).astype(np.float)
np.median(l)

# with open(os.path.join(data, "cpu.txt")) as f:
#     l = f.readlines()
# r = []
# for i in l:
#     tmp = i.split()
#     if len(tmp) > 0 and tmp[0] == "10404":
#         r.append(tmp[8])
# r = np.array(r).astype(np.float)
# np.median(r)

# with open(os.path.join(data, "gpu.log")) as f:
#     l = f.readlines()
# r = []
# for i in range(len(l)):
#     if i % 3 == 1:
#         tmp = l[i].split()
#         r.append(tmp[4])
# r = np.array(r).astype(np.float)
# np.median(r)