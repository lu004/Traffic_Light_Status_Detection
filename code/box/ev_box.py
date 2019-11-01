import cv2
import os
import numpy as np
import json
from box.i_box import i_box
from util.util import draw_b, draw_cut, iou, get_boxw, img_cut, b_fromcut

class ev_box:
    img_dir = "/demo/sig/box/img"
    label_dir = "/demo/sig/box/label"
    pred_dir = "/demo/sig/box/pred"
    shdir = "/demo/sig/box/show"

    cut = (0, 0.2, 0.6, 0.8) # y x y x
    iou_thd = 0.1

    def run(self):
        if not os.path.exists(self.pred_dir):
            os.mkdir(self.pred_dir)

        i_bx = i_box()

        for i in os.listdir(self.img_dir):
            i = i.split(".")[0]
            img = cv2.imread(os.path.join(self.img_dir, i + ".jpg"), 1)
            img2 = img_cut(img, self.cut)
            b = i_bx.run(img2)
            b = b_fromcut(img, img2, self.cut, b)
            json.dump({"file": i, "pred": b}, open(os.path.join(self.pred_dir, i + ".pred.json"), 'w'))

    def recall(self):
        r = []
        for i in os.listdir(self.label_dir):
            i = i.split(".")[0]
            img = cv2.imread(os.path.join(self.img_dir, i + ".jpg"), 1)
            gt = self.get_gt(i)
            pd = self.get_pd(i)
            for [g, g_cl] in gt:
                max = 0.0
                p_ans = None
                for [p, p_cl] in pd:
                    tmp = iou(g, p)
                    if tmp > max:
                        max = tmp
                        p_ans = p
                if max >= self.iou_thd:
                    r.append([int(get_boxw(g, img)), 1])
                else:
                    r.append([int(get_boxw(g, img)), 0])
        # count
        r = np.array(r)  # w, 1/0
        print("recall:")
        print(len(r[r[:, 1] == 1])/len(r))
        w = range(0, int(r[:, 0].max()) + 5, 5)
        print("box width", "recall", "#")
        for i in range(len(w) - 1):
            r2 = r[np.logical_and(r[:, 0] >= w[i], r[:, 0] < w[i + 1])]
            if len(r2) > 0:
                value = len(r2[r2[:, 1] == 1]) / len(r2)
                print(w[i], "{0:.3f}".format(value), len(r2))

    def prec(self):
        r = []
        for i in os.listdir(self.label_dir):
            i = i.split(".")[0]
            img = cv2.imread(os.path.join(self.img_dir, i + ".jpg"), 1)
            gt = self.get_gt(i)
            pd = self.get_pd(i)
            for [p, p_cl] in pd:
                max = 0.0
                p_ans = None
                for [g, g_cl] in gt:
                    tmp = iou(p, g)
                    if tmp > max:
                        max = tmp
                        p_ans = g
                if max >= self.iou_thd:
                    r.append([int(get_boxw(p, img)), 1])
                else:
                    r.append([int(get_boxw(p, img)), 0])

        # count
        r = np.array(r)  # w, 1/0
        print("prec:")
        print(len(r[r[:, 1] == 1])/len(r))
        w = range(0, 345, 5)

        print("box width", "prec", "#")
        for i in range(len(w) - 1):
            r2 = r[np.logical_and(r[:, 0] >= w[i], r[:, 0] < w[i + 1])]
            if len(r2) > 0:
                recall = len(r2[r2[:, 1] == 1]) / len(r2)
                print(w[i], "{0:.3f}".format(recall), len(r2))

    def show(self):
        # show
        if not os.path.exists(self.shdir):
            os.mkdir(self.shdir)
        for i in os.listdir(self.pred_dir):
            i = i.split(".")[0]
            re = self.get_pd(i)
            if len(re) > 0:
                img = cv2.imread(os.path.join(self.img_dir, i + ".jpg"), 1)
                img = draw_cut(img, self.cut)
                h, w = img.shape[0], img.shape[1]
                for [b, b_cl] in re:
                    b = np.array([[b[1] * w, b[0] * h],
                                  [b[3] * w, b[0] * h],
                                  [b[3] * w, b[2] * h],
                                  [b[1] * w, b[2] * h]])
                    img = draw_b(img, b, b_cl, color=(0, 0, 255))
                cv2.imwrite(os.path.join(self.shdir, i + ".jpg"), img)

    def get_gt(self, f):
        re = []
        d = json.load(open(os.path.join(self.label_dir, f + ".json"), "r"))
        h, w = float(d["img_h"]), float(d["img_w"])
        for i in d["box"]:
            # y x y x, class
            re.append([[i["y1"]/h, i["x1"]/w, i["y2"]/h, i["x2"]/w], i["class"]])

        # in cut
        tmp = []
        for [b, cl] in re:
            if self.cut[0] <= b[0] and self.cut[1] <= b[1] and self.cut[2] >= b[2] and self.cut[3] >= b[3]:
                tmp.append([b, cl])
        re = tmp
        return re

    def get_pd(self, f):
        re = []
        if os.path.exists(os.path.join(self.pred_dir, f + ".pred.json")):
            for i in json.load(open(os.path.join(self.pred_dir, f + ".pred.json"), "r"))["pred"]:
                b = i["b"]
                # y x y x, class
                re.append([[b[0], b[1], b[2], b[3]], i["b_cl"]])
        return re

e = ev_box()
e.run()
e.show()
e.recall()
e.prec()
