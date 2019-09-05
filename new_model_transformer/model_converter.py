""" convert old ckpt file to tensorflow pb and ckpt model"""

from __future__ import print_function
import sys
sys.path.insert(0, '../src')
import transform, numpy as np, vgg, pdb, os
import transform_sm as transform
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy
BATCH_SIZE = 4




def convert_model(old_model_dir, new_model_dir):
    with tf.compat.v1.Session() as sess:
        batch_shape = (None, None, None, 3)
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')
        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(old_model_dir):
            ckpt = tf.train.get_checkpoint_state(old_model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, old_model_dir)
        tf.saved_model.simple_save(
            sess,
            new_model_dir,
            inputs={"x": img_placeholder},
            outputs={"y": preds}
        )
    tf.compat.v1.reset_default_graph()
    return

def get_pb_model():
    models_dir = os.listdir("../data/models/")

    for i, style_name in enumerate(models_dir):
        print("process {}, {}/{}".format(style_name, i, len(models_dir)))
        old_model_dir = "../data/models/" + style_name
        new_model_dir = "./pb_models/" + style_name
        convert_model(old_model_dir, new_model_dir)

def get_ckpt_model():
    models_dir = os.listdir("temp")

    for i, style_name in enumerate(models_dir):
        style_name, _ = os.path.splitext(style_name)
        batch_shape = (None, None, None, 3)
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')
        preds = transform.net(img_placeholder)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            print("process {}, {}/{}".format(style_name, i, len(models_dir)))
            old_model_dir = "temp/" + style_name + ".ckpt"
            new_model_dir = "./temp2/" + style_name + "/"
            if os.path.isdir(old_model_dir):
                ckpt = tf.train.get_checkpoint_state(old_model_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    raise Exception("No checkpoint found...")
            else:
                saver.restore(sess, old_model_dir)
            print(saver.save(sess, new_model_dir))
        tf.compat.v1.reset_default_graph()

get_ckpt_model()


old_model_dir = "../data/models/wave"
new_model_dir = "./pb_models/wave2"






import tensorflow as tf

models_dir = os.listdir("./pb_models/")
input_shapes=[1,300,300,3]
saved_model_dir= "./pb_models/adrien_converse"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, input_shapes=[1,300,300,3])
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)