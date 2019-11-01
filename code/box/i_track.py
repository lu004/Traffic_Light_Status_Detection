import cv2
from util.util import iou

class i_track:
    ts = []
    tid = 0

    def run(self, img, b_b):

        # track
        for t in self.ts:
            ok, b = t.upd(img)
            if not ok:
                t.close()
        self.ts = [t for t in self.ts if t.run == True]

        # merge with det
        tmp = []
        for t in self.ts:
            for b in b_b:
                if iou(t.b, b) > 0.0: # overlap
                    #t.close2()
                    t.init(img, b)
                    b_b.remove(b)
                    break
        #self.ts = [t for t in self.ts if t.run == True]

        for b in b_b:
            self.tid += 1
            self.ts.append(tk(str(self.tid), img, b))

        # return
        re = []
        for t in self.ts:
            re.append(t.b)

        return re


class tk:

    def __init__(self, id, img, b):
        self.id = id
        self.run = True
        self.cc = 0 # close cnt
        self.t = cv2.TrackerKCF_create()
        #self.t = cv2.TrackerTLD_create()
        self.init(img, b)

    def init(self, img, b):
        #img = self.get_img_nb(img, b)
        self.t.init(img, self._toc2(img, b))
        self.b = b
        #print(self.id + " tk init")

    def upd(self, img):
        #img = self.get_img_nb(img, self.b)
        ok, b = self.t.update(img)
        if ok:
            self.b = self._toc1(img, b)
        return ok, b

    def close(self):
        self.cc += 1
        if self.cc >= 1:
            self.close2()

    def close2(self):
        self.run = False
        self.t.clear()
        #print(self.id + " tk close")

    def get_img_nb(self, img, b):
        # y x y x
        h, w = img.shape[0], img.shape[1]

        img2 = img[int(h*max(b[0] - 0.1, 0)):
                   int(h*min(b[2] + 0.1, 1)),
                   int(w*max(b[1] - 0.1, 0)):
                   int(w*min(b[3] + 0.1, 1))]

        cv2.imshow('img2', img2)
        return img2



    def _toc1(self, img, bx):
        h, w = img.shape[0], img.shape[1]
        bx2 = (bx[1]/h, bx[0]/w, (bx[1]+bx[3])/h, (bx[0]+bx[2])/w)
        # y x y x [0-1]
        return bx2

    def _toc2(self, img, bx):
        h, w = img.shape[0], img.shape[1]
        bx2 = (bx[1] * w, bx[0] * h, (bx[3] - bx[1]) * w, (bx[2] - bx[0]) * h)
        # x y w h
        return bx2
