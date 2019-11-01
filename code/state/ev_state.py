import os
import json
from state.ii_state import ii_state
from util.util import *

class ev_state:
    img_dir = "/demo/sig/state/img"
    pred_dir = "/demo/sig/state/pred"

    cmap = {"Green": 0, "Yellow": 1, "Red": 2, "_": 3}
    sc_thd = 0.3

    def run(self):
        if not os.path.exists(self.pred_dir):
            os.mkdir(self.pred_dir)
        ii_st = ii_state()
        r = []
        for i in os.listdir(self.img_dir):
            i = i.split(".")[0]
            img = cv2.imread(os.path.join(self.img_dir, i + ".jpg"), 1)
            r2 = ii_st.run2(img)
            r2["img"] = i
            r2["s_sc"] = float(r2["s_sc"])
            r.append(r2)
        json.dump(r, open(os.path.join(self.pred_dir, "pd_state.json"), 'w'))

    def prec(self):
        gt = np.array([0, 0, 0, 0])
        ok = np.array([0, 0, 0, 0])
        d = []
        for i in json.load(open(os.path.join(self.pred_dir, "pd_state.json"), 'r')):
            g = i["img"].split("-")[-1]
            p = i["s"]
            sc = i["s_sc"]
            img = cv2.imread(os.path.join(self.img_dir, i["img"]+".jpg"), 1)
            
            gt[self.cmap[g]] += 1
            if g == p and sc >= self.sc_thd:
                ok[self.cmap[g]] += 1
                d.append([self.cmap[g], img.shape[1], 1])
            else:
                d.append([self.cmap[g], img.shape[1], 0])
        print(self.cmap)
        print("prec(all): " + str(ok.sum()/gt.sum()))
        print("prec: " + str(ok/gt))

        # prec
        d = np.array(d)
        w = range(0, int(d[:, 1].max()) + 5, 5)
        print("\n" + "color", "box_width", "prec", "#")
        for cl in [0, 1, 2, 3]:
            for i in range(len(w)-1):
                d2 = d[np.logical_and(d[:, 0] == cl, np.logical_and(d[:, 1] >= w[i], d[:, 1] < w[i+1]))]
                if len(d2) > 0:
                    pre = len(d2[d2[:, 2] == 1])/len(d2)
                    print(cl, w[i], "{0:.2f}".format(pre), len(d2))

        # gt -> pd
        d = np.array([[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]])
        for i in json.load(open(os.path.join(self.pred_dir, "pd_state.json"), 'r')):
            g = i["img"].split("-")[-1]
            p = i["s"]
            sc = i["s_sc"]
            gt[self.cmap[g]] += 1
            d[self.cmap[g], self.cmap[p]] += 1

        print("\n ground truth -> pred: " + str(self.cmap))
        print(d)

e = ev_state()
e.run()
e.prec()
