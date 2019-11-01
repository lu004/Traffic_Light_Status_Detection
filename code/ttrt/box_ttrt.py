import os
import numpy as np
import time
import cv2
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

class pd_box:
    grf_file = "/box/model/ssd_3.3/frozen_inference_graph.pb"
    prec_mode = "FP16"
    img_h, img_w = 640, 640
    input_nd_str = "image_tensor"
    out_nd_str = ["detection_boxes", "detection_scores", "detection_classes", "num_detections"]
    model_id = grf_file.split("/")[-2]
    input_nd = None
    out_nd = []
    tf_sess = None
    cmap = {1: "red", 2: "yellow", 3: "blue", 4: "none"}
    conf_thd = 0.5

    def __init__(self):
        g = tf.Graph()
        with g.as_default():
            #tf.import_graph_def(self._load_grf(self.grf_file), name='')
            tf.import_graph_def(self.get_trt(self._load_grf(self.grf_file), self.prec_mode), name='') # trt

            # for tmp in g.get_operations():
            #     print(tmp.name)
            # input, output nd

            self.input_nd = g.get_tensor_by_name(self.input_nd_str + ':0')
            for i in range(len(self.out_nd_str)):
                self.out_nd.append(g.get_tensor_by_name(self.out_nd_str[i] + ':0'))
            self.tf_sess = tf.Session(graph=g)


    def run(self, img):
        # need img: [1, h, w, 3:RGB]
        img = self._set_img(img)
        stime = time.time()
        (box, sc, cl, num) = self.tf_sess.run([self.out_nd[0], self.out_nd[1], self.out_nd[2], self.out_nd[3]],
                                              feed_dict={self.input_nd: img})
        print("pd_box: " + str(time.time() - stime))
        return self._get_rs(np.squeeze(box), np.squeeze(sc), np.squeeze(cl).astype(np.int32))



    def _set_img(self, img):
        size = (self.img_w, self.img_h) # w h
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR) # resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        return img

    def _get_rs(self, bx, sc, cl):
        tmp = sc > self.conf_thd
        bx, sc, cl = bx[tmp], sc[tmp], cl[tmp]
        out = []
        if bx.size > 0:
            for i in range(len(bx)):
                out.append({"box": bx[i].tolist(), "score": sc.tolist()[i], "class": self.cmap[cl[i]]})
        return out

    def get_trt(self, grf, precision):
        re = trt.create_inference_graph(grf, [self.out_nd_str[0], self.out_nd_str[1], self.out_nd_str[2], self.out_nd_str[3]],
                                             max_batch_size=20,
                                             max_workspace_size_bytes=2<<10 << 20,
                                             precision_mode=precision,
                                             minimum_segment_size=10)
        return re

    def get_trt_from_calib(self, grf_calib):
        """Convert a TensorRT graph used for calibration to an inference graph."""
        grf_trt = trt.calib_graph_to_infer_graph(grf_calib)
        return grf_trt

    def _load_grf(self, grf_file):
        with tf.gfile.GFile(grf_file, 'rb') as f:
            grf_def = tf.GraphDef()
            grf_def.ParseFromString(f.read())
        return grf_def

    def _store_grf(self, grf_name, grf):
        output_path = os.path.join(self.outdir, grf_name)
        with tf.gfile.GFile(output_path, "wb") as f:
            f.write(grf.SerializeToString())


img_dir = '/sigdata/201812/box/test/img'
pdb = pd_box()
for f in os.listdir(img_dir):
    img = cv2.imread(os.path.join(img_dir, f), 1)
    bx = pdb.run(img)
    if len(bx) > 0:
        print(bx[0]["box"])
    stp = 0
