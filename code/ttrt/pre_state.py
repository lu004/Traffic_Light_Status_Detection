import tensorflow as tf
import numpy as np

class pre_state:
    central = 0.875

    def run(self, img, h, w):
        #img = tf.image.decode_jpeg(tf.read_file(img_pth), channels=3)
        img = self.set_img(img, h, w)
        return img

    def set_img(self, img, h, w):
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.expand_dims(img, 0)
        s = (1 - self.central) / 2
        e = 1 - s
        img = tf.image.crop_and_resize(img, [[s, s, e, e]], [0], [h, w])
        img = tf.squeeze(img, [0])
        img = tf.subtract(img, 0.5)
        img = tf.multiply(img, 2.0)
        img = tf.expand_dims(img, axis=0)
        return img

    def set_img2(self, img, h, w):
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.central_crop(img, central_fraction=self.central)
        img = tf.expand_dims(img, 0)
        img = tf.image.resize_bilinear(img, [h, w], align_corners=False)
        img = tf.squeeze(img, [0])
        img = tf.subtract(img, 0.5)
        img = tf.multiply(img, 2.0)
        img = tf.expand_dims(img, axis=0)
        return img
