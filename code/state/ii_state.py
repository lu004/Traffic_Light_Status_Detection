import numpy as np
import tensorflow as tf
import cv2
from util.util import img_cut

class ii_state:
    grf_file = "/state/model/mb1.1/mobilenet_v2_140_freeze.pb"
    id = grf_file.split("/")[-2]
    img_h, img_w = 30, 70
    #img_h, img_w = 112, 224
    input_nd_str = "input"
    output_nd_str = "MobilenetV2/Predictions/Reshape_1"
    #output_nd_str = "InceptionV2/Predictions/Reshape_1"
    input_nd = None
    output_nd = None
    img_nd = None
    img2_nd = None
    tf_sess = None
    cmap = {0: "Green", 1: "Yellow", 2: "Red", 3: "_"}
    sc_thd = 0.8

    def __init__(self):
        g = tf.Graph()
        with g.as_default():
            tf.import_graph_def(self._load_grf(self.grf_file), name='')
            self.input_nd = g.get_tensor_by_name(self.input_nd_str + ':0')
            self.output_nd = g.get_tensor_by_name(self.output_nd_str + ':0')
            self.img_nd = tf.placeholder(tf.uint8, shape=(None, None, 3))  # h, w, 3
            self.img2_nd = pre_st().run(self.img_nd, self.img_h, self.img_w)
            self.tf_sess = tf.Session(graph=g)


    def run(self, img_f, b):
        #stime = time.time()
        re = []
        for i in range(len(b)):
            b2 = b[i]["b"]
            img = img_cut(img_f, (b2[0]-0.01, b2[1]-0.01, b2[2]+0.01, b2[3]+0.01))
            if img.size > 0:
                #cv2.imshow(str(i), img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img2 = self.tf_sess.run(self.img2_nd, feed_dict={self.img_nd: img})
                out = self.tf_sess.run(self.output_nd, feed_dict={self.input_nd: img2})
                re.append({"s": self.cmap[np.argmax(out)],"s_sc": np.max(out)})
            else:
                re.append({"s": "none", "s_sc": -1.0})

        return re

    def run2(self, img):
        if img.size > 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img2 = self.tf_sess.run(self.img2_nd, feed_dict={self.img_nd: img})
            out = self.tf_sess.run(self.output_nd, feed_dict={self.input_nd: img2})
            return {"s": self.cmap[np.argmax(out)], "s_sc": np.max(out)}
        else:
            return {"s": "none", "s_sc": -1.0}


    def _load_grf(self, grf_file):
        with tf.gfile.GFile(grf_file, 'rb') as f:
            grf_def = tf.GraphDef()
            grf_def.ParseFromString(f.read())
        return grf_def



class pre_st:
    #central = 0.875
    central = 0.95
    def run(self, img, h, w):
        img = self._set_img(img, h, w)
        return img

    def _set_img(self, img, h, w):
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.central_crop(img, central_fraction=self.central)
        img = tf.expand_dims(img, 0)
        img = tf.image.resize_bilinear(img, [h, w], align_corners=False)
        img = tf.squeeze(img, [0])
        img = tf.subtract(img, 0.5)
        #img = tf.multiply(img, 2.0)
        img = tf.multiply(img, 1.0)
        img = tf.expand_dims(img, axis=0)
        return img

    # def _set_img(self, img, h, w):
    #     img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    #     img = tf.expand_dims(img, 0)
    #     s = (1 - self.central) / 2
    #     e = 1 - s
    #     img = tf.image.crop_and_resize(img, [[s, s, e, e]], [0], [h, w])
    #     img = tf.squeeze(img, [0])
    #     img = tf.subtract(img, 0.5)
    #     img = tf.multiply(img, 2.0)
    #     img = tf.expand_dims(img, axis=0)
    #     return img

