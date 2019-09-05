import time
st = time.time()
import tensorflow as tf
import scipy.misc, numpy as np, os, sys
from PIL import Image

def get_img(src, img_size=False):
    #img = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
    img = np.asarray(Image.open(src))
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img,img,img))
    if img_size != False:
        img = np.array(Image.fromarray(img).resize(img_size))
    return img

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    #scipy.misc.imsave(out_path, img)
    Image.fromarray(img).save(out_path)


config = tf.ConfigProto(device_count = {'GPU': 0})
sess = tf.Session(config=config)
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
input_key = 'x'
output_key = 'y'


st = time.time()
export_path =  './new_models/udnie'
meta_graph_def = tf.saved_model.loader.load(
         sess,
        [tf.saved_model.tag_constants.SERVING],
        export_path)

signature = meta_graph_def.signature_def

x_tensor_name = signature[signature_key].inputs[input_key].name
y_tensor_name = signature[signature_key].outputs[output_key].name

x = sess.graph.get_tensor_by_name(x_tensor_name)
y = sess.graph.get_tensor_by_name(y_tensor_name)

x_input = get_img("1565956857186.jpg")
x_input = np.expand_dims(x_input, 0)
y_out = sess.run(y, feed_dict={x: x_input})
save_img("temp123.jpg", y_out[0])
print(time.time() - st)


if __name__ == '__main__':
    main()
    print(time.time() - st)
