import os
import numpy as np
import time
import json
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import cv2
from ttrt.pre_state import pre_state

class pd_state:
    img_dir = "/sigdata/201812/state/test"
    grf_file = "/state/model/mobilenet/mobilenet_v2_140_freeze.pb"
    input_nd = "input"
    output_nd = "MobilenetV2/Predictions/Reshape_1"
    img_h = 112
    img_w = 224
    cmap = {0: "red", 1: "yellow", 2: "blue", 3: "none"}
    outdir = "/exp/state"

    def run(self):
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

        grf = self._load_grf(self.grf_file)
        self.run2(grf, "")

        fp32 = self.get_trt(grf, "FP32")
        self.run2(fp32, "FP32")

        # fp16 = self.get_trt(grf, "FP16")
        # self.run2(fp16, "FP16")
        #
        # int8 = self.get_trt(grf, "INT8")
        # self.run2(int8, "INT8")
        #
        # int8_calib = self.get_trt_from_calib(int8)
        # self.run2(int8_calib, "INT8_calib")


    def run2(self, grf, mode):
        rtime = []
        re = {}
        g = tf.Graph()
        with g.as_default():
            # grf, input node, output node
            tf.import_graph_def(grf, name='')
            input = g.get_tensor_by_name(self.input_nd+":0")
            output = g.get_tensor_by_name(self.output_nd+":0")
            img = tf.placeholder(tf.uint8, shape=(None, None, 3))
            pre = pre_state().run(img, self.img_h, self.img_w)
            with tf.Session(graph=g) as sess:
                for f in os.listdir(self.img_dir):
                    # img: [1, h, w, 3:RGB]
                    tmp = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, f), 1), cv2.COLOR_BGR2RGB)

                    img_pre = sess.run(pre, feed_dict={img: tmp})
                    stime = time.time()
                    output_re = sess.run(output, feed_dict={input: img_pre})
                    rtime.append((time.time() - stime))
                    # store
                    cl, sc = self.cmap[np.argmax(output_re)], np.max(output_re)
                    re[f.split(".")[0]] = cl+"_sc_"+"{0:.2f}".format(sc)

        with open(os.path.join(self.outdir, "pdstate_"+mode+".json"), 'w') as f:
            json.dump(re, f)

        print(mode + " median time: " + str(np.median(rtime)))

    def get_trt(self, grf, precision):
        re = trt.create_inference_graph(grf, [self.output_nd],
                                             max_batch_size=128,
                                             max_workspace_size_bytes=2<<10 << 20,
                                             precision_mode=precision)
        return re

    def get_trt_from_calib(self, grf_calib):
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

pd_state().run()
