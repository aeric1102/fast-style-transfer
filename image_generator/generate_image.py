import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import os
import numpy as np
import time
import socket
import sys
import gc
from PIL import Image

import sys
sys.path.insert(0, '../src')
import transform, numpy as np, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.compat.v1.ConfigProto()
def get_img(src, img_size=False):
    img = np.asarray(Image.open(src))
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img,img,img))
    if img_size != False:
        img = np.array(Image.fromarray(img).resize(img_size))
    return img

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)


def convert(model_path, input_paths, output_paths):
    sess = tf.Session(config=config)
    batch_shape = (None, None, None, 3)
    img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                     name='img_placeholder')
    preds = transform.net(img_placeholder)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    for input_path, output_path in zip(input_paths, output_paths):
        # Transform image
        x_input = get_img(input_path)
        x_input = np.expand_dims(x_input, 0)
        y_out = sess.run(preds, feed_dict={img_placeholder: x_input})
        save_img(output_path, y_out[0])
    tf.compat.v1.reset_default_graph()
    gc.collect()
    return


import os
import random
import numpy as np
import uuid
import json
import shutil

img_per_model = 30
models_dir = "./ckpt_models/"
images_dir = "../data/train2014/"

output_dir = "./data/outputs/"
styles = os.listdir(models_dir)
model_paths = [models_dir + s + "/" for s in styles]
n_model = len(model_paths)
n_image = n_model * img_per_model
image_paths = random.sample(os.listdir(images_dir), n_image)
image_paths = [images_dir + p for p in image_paths]
# store selected image in a new directory
for img in image_paths:
    filename = os.path.basename(img)
    shutil.copy2(img, "./data/contents/"+filename)

image_paths = ["./data/contents/" + os.path.basename(img) for img in image_paths]
output_paths = [output_dir + uuid.uuid4().hex + ".jpg" for i in range(n_image)]






cur_count = 0
for i, model_path in enumerate(model_paths):
    cur_input_paths = image_paths[i*img_per_model:(i+1)*img_per_model]
    cur_output_paths = output_paths[i*img_per_model:(i+1)*img_per_model]    
    convert(model_path, cur_input_paths, cur_output_paths)
    print(i, len(model_paths), model_path)

total_styles = []
for s in styles:
    total_styles.extend([s]*img_per_model)

output = []
for i in range(n_image):
    post = dict({
            "contentImg": image_paths[i],
            "selectStyle": total_styles[i],
            "resultImg": output_paths[i],
        })
    output.append(post)


with open("mapping.json", "w") as outfile:
    json.dump(output, outfile)









# transform one image in temp dir
img_per_model = 30
models_dir = "./ckpt_models/"
output_dir = "./temp3/"
styles = os.listdir(models_dir)
model_paths = [models_dir + s + "/" for s in styles]
images_dir = "./temp/"
image_paths = os.listdir(images_dir)
image_paths = [images_dir + p for p in image_paths]
output_paths = [output_dir + styles[i] + ".jpg" for i in range(len(styles))]

cur_count = 0
for i, model_path in enumerate(model_paths):
    cur_input_paths = [image_paths[0]]
    cur_output_paths = [output_paths[i]]    
    convert(model_path, cur_input_paths, cur_output_paths)
    print(i, len(model_paths), model_path)