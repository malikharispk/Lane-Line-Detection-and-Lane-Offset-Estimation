import os

import tensorflow as tf
import numpy as np
from skimage import io

from Dataset import ImageReader
import FCN
import cv2
import time
# Define training settings.
NUM_OF_CLASSESS = 2
BATCH_SIZE = 2
LEARNING_RATE = 1e-4

# Define directories of checkpoint, data and model.
LOGS_DIR = "logs"
DATA_DIR = "Data_zoo/MIT_SceneParsing"
MODEL_DIR = "MODEL_ZOO"

# Define image and output directory.
IMAGE_DIR = "infer"
OUTPUT_DIR = "output"


def metrics(label: np.ndarray, pred: np.ndarray, num_class: np.uint8):
    label = label.astype(np.uint8)
    pred = np.round(pred)
    mat = np.zeros((num_class, num_class))
    [height, width] = label.shape
    for i in range(height):
        for j in range(width):
            pixel_label = label[i][j]
            pixel_pred = pred[i][j]
            mat[pixel_label][pixel_pred] += 1

    total_n_jis = np.sum(mat, axis=0)
    total_is = np.sum(mat, axis=1)

    ious = []  # a list of intersection over union for each i
    t_n_ii = []  # the total number of pixels of class i that is correctly predicted
    total_iou_n_ii = []  # a list of (iou * n_ii) for each i
    total_n_ii_divied_t_i = []
    for i in range(1, num_class):  # calculate iou for class in [1, num_class]
        n_ii = mat[i][i]  # the number of pixels of class i that is correctly predicted

        total_i = total_is[i]  # the total number of pixels of class i
        total_n_ji = total_n_jis[i]  # the total number of pixels predicted to be class i
        if total_i == 0:
            continue

        t_n_ii.append(n_ii)
        total_n_ii_divied_t_i.append(n_ii * 1.0 / total_i)

        if n_ii == 0:
            iou = 0 # intersection over union
        else:
            iou = (n_ii + 0.0) / (total_i + total_n_ji - n_ii)
        total_iou_n_ii.append(iou * total_i)
        ious.append(iou)
    # print(ious)

    pixel_acc = np.sum(t_n_ii) / np.sum(mat)
    mean_acc = np.sum(total_n_ii_divied_t_i) / len(t_n_ii)
    mean_intersection_over_union = np.mean(np.array(ious))
    frequency_weighted_iu = np.sum(total_iou_n_ii) * 1.0 / np.sum(mat)

    return pixel_acc, mean_acc, mean_intersection_over_union, frequency_weighted_iu


def main(argv=None):
    # Define tensors.
    image_width = 352
    image_height = 160
    # Define loss tensor operations.
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")
    images = tf.placeholder(tf.float32, shape=[None, image_height, image_width, 3], name="input_image")
    annotations = tf.placeholder(tf.int32, shape=[None, image_height, image_width, 1], name="annotations")
    pred_annotation, logits, pre_offset = FCN.deeplab_largfov(images, keep_probability, NUM_OF_CLASSESS, 1)

    # Define image reader.
    print("setting up ImageReader...")
    image_reader = ImageReader(IMAGE_DIR)

    # Define session config, init session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Initialize variables.
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore model.
    print("Setting up Saver...")
    saver = tf.train.Saver()
    step = 1  # start_step
    ckpt = tf.train.latest_checkpoint(LOGS_DIR)
    if ckpt:
        saver.restore(sess, ckpt)
        print("Model restored...")
        print("restore from the checkpoint {0}".format(ckpt))
        print(ckpt.split('-'))
        step += int(ckpt.split('-')[-1])
    print(step)

    while image_reader.has_next:
        # Session Calculate predicted annotation.
        start_time = time.time()
        image = image_reader.next_image()
        index = image_reader.cur_index
        fd = {images: image, keep_probability: 1.0}
        preds = sess.run(pred_annotation, feed_dict=fd)
        speed_time = time.time() - start_time
        print("fps", 1 / speed_time)
        print(preds.shape)
        pred = np.squeeze(preds[0], axis=2)
        image = np.squeeze(image, axis=0)
        # Save prediction as file.
        output_path = os.path.join(OUTPUT_DIR, "prediction_" + str(index) + ".png")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
        mask = cv2.dilate(pred.astype(np.uint8)*100, kernel)
        io.imsave(output_path, mask)
        image_path = os.path.join(OUTPUT_DIR, "image_" + str(index) + ".png")
        mask = cv2.merge([pred.astype(np.uint8)*200,pred.astype(np.uint8)*200,pred.astype(np.uint8)*200])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
        mask = cv2.dilate(mask,kernel)
        image = image + mask
        io.imsave(image_path, image)
        print("Saved image: %s" % output_path)



if __name__ == '__main__':
    tf.app.run()
