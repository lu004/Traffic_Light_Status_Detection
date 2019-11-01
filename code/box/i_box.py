import numpy as np
import tensorflow as tf
import time
import cv2
g_mem = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

class i_box:
    grf_file = "/box/model/ssd1.0/frozen_inference_graph.pb"
    id = grf_file.split("/")[-2]
    img_h, img_w = 640, 640
    #img_h, img_w = 800, 800
    #img_h, img_w = 600, 1143
    input_nd_str = "image_tensor"
    output_nd_str = ["detection_boxes", "detection_scores", "detection_classes", "num_detections"]
    input_nd = None
    output_nd = []
    tf_sess = None
    cmap = {1: "Green", 2: "Yellow", 3: "Red", 4: "_"}
    #cmap = {1: "Red", 2: "Yellow", 3: "Green", 4: "_"}
    #cmap = {1: "box"}
    sc_thd = 0.5

    def __init__(self):
        print(self.id)
        g = tf.Graph()
        with g.as_default():
            tf.import_graph_def(self._load_grf(self.grf_file), name='')
            self.input_nd = g.get_tensor_by_name(self.input_nd_str + ':0')
            for i in range(len(self.output_nd_str)):
                self.output_nd.append(g.get_tensor_by_name(self.output_nd_str[i] + ':0'))
            self.tf_sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=g_mem))

    def run(self, img):
        # need img: [1, h, w, 3:RGB]
        img = self._set_img(img)
        stime = time.time()
        (box, sc, cl, num) = self.tf_sess.run([self.output_nd[0], self.output_nd[1], self.output_nd[2], self.output_nd[3]],
                                              feed_dict={self.input_nd: img})
        #qprint("pd_box: " + str(time.time() - stime))
        return self._get_re(np.squeeze(box), np.squeeze(sc), np.squeeze(cl))

    def _set_img(self, img):
        size = (self.img_w, self.img_h)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR) # resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        return img

    def _get_re(self, bx, sc, cl):
        tmp = sc > self.sc_thd
        bx, sc = bx[tmp], sc[tmp]
        out = []
        if bx.size > 0:
            for i in range(len(bx)):
                out.append({"b": bx[i].tolist(), "b_sc": sc.tolist()[i], "b_cl": self.cmap[cl[i]]})
        return out

    def _load_grf(self, grf_file):
        with tf.gfile.GFile(grf_file, 'rb') as f:
            grf_def = tf.GraphDef()
            grf_def.ParseFromString(f.read())
        return grf_def
