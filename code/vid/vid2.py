import cv2
import numpy as np
import os
from natsort import natsorted
from box.i_box import i_box
from state.ii_state import ii_state
from util.util import draw_b, draw_cut, img_cut

def draw_re(img, img2, c, re):
    h, w = img.shape[0]*c[0], img.shape[1]*c[1]
    h2, w2 = img2.shape[0], img2.shape[1]
    for r in re:
        b = r["b"]
        b = np.array([[b[1] * w2 + w, b[0] * h2 + h],
                      [b[3] * w2 + w, b[0] * h2 + h],
                      [b[3] * w2 + w, b[2] * h2 + h],
                      [b[1] * w2 + w, b[2] * h2 + h]])
        if r["s"] == "Red":
            img = draw_b(img, b, "R" + "_" + "{0:.2f}".format(r["s_sc"]), color=(0, 0, 255))
        elif r["s"] == "Yellow":
            img = draw_b(img, b, "Y" + "_" + "{0:.2f}".format(r["s_sc"]), color=(0, 0, 255))
        elif r["s"] == "Green":
            img = draw_b(img, b, "G" + "_" + "{0:.2f}".format(r["s_sc"]), color=(255, 0, 0))
        elif r["s"] == "_":
            img = draw_b(img, b, "_" + "_" + "{0:.2f}".format(r["s_sc"]), color=(125, 125, 125))
    return img

def get_final_s(re):
    c = {}
    sc = {}
    for r in re:
        c[r["s"]] = c[r["s"]] +1 if r["s"] in c else 1
        sc[r["s"]] = sc[r["s"]] + r["s_sc"] if r["s"] in sc else r["s_sc"]
    ans = max(c, key=c.get)
    ans_c = max(c.values())
    for k, v in sc.items():
        if c[k] == ans_c:
            return max(sc, key=sc.get)
    return ans


i_bx = i_box()
#i_trk = i_track()
ii_st = ii_state()
#cut = (0, 1, 0, 1) # y x y x
cut = (0.0, 0.2, 0.6, 0.8) # y x y x

img_dir = "/exp/2/test/img2"
vid_w = int(1920*0.25)
vid_h = int(1200*0.25)
fps = 7.0
vidr = cv2.VideoWriter("/exp/2/"+"cut_"+i_bx.id+"_"+ii_st.id+".mp4",
                       cv2.VideoWriter_fourcc(*'XVID'), fps, (vid_w, vid_h))


lst = natsorted(os.listdir(img_dir))
for t in range(len(lst)):
    if t % 1 == 0:
        print(t)
        img = cv2.imread(os.path.join(img_dir, lst[t]), 1)
        img2 = img_cut(img, cut)

        b = i_bx.run(img2)
        #b = i_trk.run(img2, b)

        s = ii_st.run(img2, b)
        re = []
        for i in range(len(b)):
            re.append({"b": b[i]["b"], "s": s[i]["s"], "s_sc": s[i]["s_sc"]})

        fs = ""
        if len(re) > 0:
            fs = get_final_s(re)

        # show
        img = draw_cut(img, cut)
        if len(re) > 0:
            img = draw_re(img, img2, cut, re)
        if fs == "Green":
            cv2.putText(img, "Green" + " (this frame)", (int(img.shape[1]/4), 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 4, cv2.LINE_AA)
        elif fs == "Red" or fs == "Yellow":
            cv2.putText(img, "Red/Yellow" + " (this frame)", (int(img.shape[1]/4), 70), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 4, cv2.LINE_AA)

        img = cv2.resize(img, (vid_w, vid_h))
        vidr.write(img)
