from shapely.geometry import Polygon
import math
import numpy as np
import cv2

def draw_b(img, b, text=None, color=(0, 255, 0), tk=4):
    if len(b) == 4:
        b = b.astype(int)
        p = np.r_[b[:4], b[0].reshape(-1,2)]
        for i in range(len(p)-1):
            cv2.line(img, (p[i, 0], p[i, 1]),(p[i+1, 0], p[i+1, 1]), color, tk)
        if text is not None:
            cv2.putText(img, text, (min(b[:, 0]), min(b[:, 1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
    return img

def draw_cut(img, c): # y x y x
    h, w = img.shape[0], img.shape[1]
    b = np.array([[c[1] * w, c[0] * h],
                  [c[3] * w, c[0] * h],
                  [c[3] * w, c[2] * h],
                  [c[1] * w, c[2] * h]])
    img = draw_b(img, b, "CUT", color=(0, 0, 0))
    return img

def img_cut(img, c): # y x y x
    h, w = img.shape[0], img.shape[1]
    img2 = img[int(h*c[0]):int(h*c[2]), int(w*c[1]):int(w*c[3])]
    return img2

def b_fromcut(img, img2, c, b):
    # c y x y x
    # b: y x y x
    h, w = img.shape[0], img.shape[1]
    h2, w2 = img2.shape[0], img2.shape[1]
    for r in b:
        r = r["b"]
        r[0] = c[0] + r[0] * h2/h
        r[1] = c[1] + r[1] * w2/w
        r[2] = c[0] + r[2] * h2/h
        r[3] = c[1] + r[3] * w2/w
    return b

def adj_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def iou(a, b):
    # y x y x
    p1 = Polygon([(a[1], a[0]), (a[3], a[0]), (a[3], a[2]), (a[1], a[2])])
    p2 = Polygon([(b[1], b[0]), (b[3], b[0]), (b[3], b[2]), (b[1], b[2])])
    int = p1.intersection(p2).area
    uni = float(p1.area+p2.area-int)
    return int/uni if int > 0.0 else 0.0

def get_boxw(b, img):
    # y x y x [0-1]
    w = img.shape[1]
    return math.fabs((b[3] - b[1]) * w)

def get_boxw2(b):
    # y x y x
    return math.fabs((b[3] - b[1]))

def get_boxh(b, img):
    # y x y x [0-1]
    h = img.shape[0]
    return math.fabs((b[2] - b[0]) * h)

def get_boxh2(b):
    # y x y x
    return math.fabs((b[2] - b[0]))
