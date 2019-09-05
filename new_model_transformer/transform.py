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

def transform1(model_paths, input_path, output_dir):
    # Load model
    st = time.time()
    for i, model_path in enumerate(model_paths):
        with tf.Session(config=config) as sess:
            meta_graph_def = tf.saved_model.loader.load(
                sess,
                [tf.saved_model.SERVING],
                model_path)
            signature = meta_graph_def.signature_def
            signature_key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            input_key = 'x'
            output_key = 'y'
            x_tensor_name = signature[signature_key].inputs[input_key].name
            y_tensor_name = signature[signature_key].outputs[output_key].name
            x = sess.graph.get_tensor_by_name(x_tensor_name)
            y = sess.graph.get_tensor_by_name(y_tensor_name)
            # Transform image
            x_input = get_img(input_path)
            x_input = np.expand_dims(x_input, 0)
            y_out = sess.run(y, feed_dict={x: x_input})
            save_img(output_dir+str(i)+".jpg", y_out[0])
            print(i, time.time()-st)
        tf.compat.v1.reset_default_graph()
        gc.collect()
    print(time.time()-st)
    return




def transform2(model_paths, input_path, output_dir):
    sess = tf.Session(config=config)
    batch_shape = (None, None, None, 3)
    img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                     name='img_placeholder')
    preds = transform.net(img_placeholder)
    saver = tf.train.Saver()
    # Load model
    st = time.time()
    for i, model_path in enumerate(model_paths):
        saver.restore(sess, model_path)
        # Transform image
        x_input = get_img(input_path)
        x_input = np.expand_dims(x_input, 0)
        y_out = sess.run(preds, feed_dict={img_placeholder: x_input})
        save_img(output_dir+str(i)+".jpg", y_out[0])
        print(i, time.time()-st)

    tf.compat.v1.reset_default_graph()
    gc.collect()
    print(time.time()-st)
    return




import time
def main():
    input_path = "./data/test.jpg"
    model_paths1 = ["./pb_models/"+d for d in os.listdir("./pb_models")]
    model_paths2 = ["./ckpt_models/"+d+"/" for d in os.listdir("./ckpt_models")]
    #transform1(model_paths1, input_path, "./data/output1/")
    transform2(model_paths2, input_path, "./data/output2/")





if __name__ == '__main__':
    main()


