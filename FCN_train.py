import datetime
import os
import numpy as np
import time
import math
import tensorflow as tf

from skimage import io
from Dataset import TrainDataset, TestDataset
import FCN
from FCN_down_sizing import metrics
from sklearn.metrics import confusion_matrix,mean_squared_error
import cv2

# Define training setting.
MAX_STEPS = int(100001)
NUM_OF_CLASSESS = 5
BATCH_SIZE = 20
LEARNING_RATE = 2e-4
LEARNING_RATE_OFF = 1e-4

# Define directories of checkpoint, data and model.
LOGS_DIR = "logs"
DATA_DIR = "data1"
MODEL_DIR = "MODEL_ZOO/"
OUTPUT_DIR = "output"
#model: pspnet/fcn_8s/pspnet_2/pspnet_res
#model = 'fcn_8s'
# model = 'tan_net'
model = 'pspnet_res'

def evaluating_indicator(pre,gt,img,name_i):
    pre = pre.astype(np.uint8)
    gt = gt.astype(np.uint8)
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # cv2.imwrite("./img/" + str(name_i) + "img.jpg", img)
    img1 = pre
    img1 = np.bitwise_not(img1)
    # cv2.imwrite("./img/" + str(name_i) + "pre.jpg", img1 * 50)
    ret, img1 = cv2.threshold(img1, 254, 255, cv2.THRESH_BINARY)

    img1 = np.bitwise_not(img1)/255
    img1 = np.reshape(img1, (160 * 352, 1))
    img2 = gt
    img2 = np.bitwise_not(img2)
    # cv2.imwrite("./img/" + str(name_i) + "gt.jpg", img2 * 50)
    ret, img2 = cv2.threshold(img2, 254, 255, cv2.THRESH_BINARY)


    img2 = np.bitwise_not(img2)/255
    img2 = np.reshape(img2, (160 * 352, 1))
    tn, fp, fn, tp = confusion_matrix(img1, img2, labels=[0,1]).ravel()
    mse = mean_squared_error(img1, img2)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    beta = 1
    F_measure = (1 + beta * beta) * ((precision * recall) / ((beta * beta * precision) + recall))
    Accuracy = (tp + tn) / (tp + fn + fp + tn)
    print('mse', mse,'Recall', recall,'precision', precision,'F_measure', F_measure,'Accuracy', Accuracy)
    print(tn, fp, fn, tp)

    return mse, recall, precision, F_measure, Accuracy

def train(loss_val, var_list,learn_rate):
    """
    Define the optimization layer.
    :return: Tensor. The tensor of training operation.
    """
    optimizer = tf.train.AdamOptimizer(learn_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    image_width = 352
    image_height = 160
    # Define loss tensor operations.
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")
    images = tf.placeholder(tf.float32, shape=[None, image_height, image_width, 3], name="input_image")
    annotations = tf.placeholder(tf.int32, shape=[None, image_height, image_width, 1], name="annotations")
    offsets = tf.placeholder(tf.int32, shape=[None, 1], name="offsets")
    if model == 'pspnet':
        print('train the pspnet')
        pred_annotation, logits = FCN.pspnet(images, keep_probability, NUM_OF_CLASSESS)
    elif model == 'pspnet_2':
        print('train the pspnet_2')
        pred_annotation, logits = FCN.pspnet_2(images, keep_probability, NUM_OF_CLASSESS)
    elif model == 'pspnet_res':
        print('train the pspnet_res')
        pred_annotation, logits, logits3, logits2, logits1, pre_offset = FCN.pspnet_res(images, keep_probability, NUM_OF_CLASSESS)
    elif model == 'fcn_32s':
        print('train the fcn_32s')
        pred_annotation, logits = FCN.get_fcn_32s_net(images, keep_probability, NUM_OF_CLASSESS)
    elif model == 'tan_net':
        print('train the tan_net')
        pred_annotation, logits,pre_offset = FCN.tan3_net(images, keep_probability, NUM_OF_CLASSESS,4)
    elif model == 'deng_model':
        pred_annotation, logits = FCN.deng_net(images,keep_probability)
    elif model == 'deeplab_largfov':
        pred_annotation, logits, pre_offset = FCN.deeplab_largfov(images, keep_probability, NUM_OF_CLASSESS,8)
    else:
        pred_annotation, logits = FCN.get_fcn_32s_net(images, keep_probability, NUM_OF_CLASSESS)
    tf.summary.image("input_image", images, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotations*200, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation*200, tf.uint8), max_outputs=2)
    # loss = tf.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.cast(logits,tf.float32),
    #                                                                       labels=tf.cast(tf.ones_like(annotations),tf.float32))))
    # loss = tf.sqrt(tf.reduce_mean(tf.square(tf.cast(annotations,tf.float32)-tf.cast(logits,tf.float32))))
    seg_loss4 = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotations,
                                                                                            squeeze_dims=[3]),
                                                                          name="entropy")))
    seg_loss3 = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits3,
                                                                               labels=tf.squeeze(annotations,
                                                                                                 squeeze_dims=[3]),
                                                                               name="entropy")))
    seg_loss2 = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2,
                                                                               labels=tf.squeeze(annotations,
                                                                                                 squeeze_dims=[3]),
                                                                               name="entropy")))
    seg_loss1 = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1,
                                                                               labels=tf.squeeze(annotations,
                                                                                                 squeeze_dims=[3]),
                                                                               name="entropy")))
                                                                               # Define train tensor operations.
    # loss_offsets = tf.sqrt(tf.reduce_mean(tf.square(tf.cast(pre_offset, tf.float32) - tf.cast(offsets, tf.float32))))
    loss_offsets = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(offsets, 7),logits=pre_offset)
    loss_offsets = tf.reduce_mean(loss_offsets)
    reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))


    seg_loss = 0.8*seg_loss4 + 0.25*seg_loss3 + 0.10*seg_loss2 + 0.05*seg_loss1
    loss_offsets = loss_offsets
    loss = seg_loss + loss_offsets
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('seg_loss', seg_loss)
    tf.summary.scalar('loss_offsets', loss_offsets)

    trainable_var_seg = tf.trainable_variables()
    train_op_seg = train(seg_loss, trainable_var_seg,LEARNING_RATE)

    trainable_var_off = tf.trainable_variables()
    train_op_off = train(loss_offsets, trainable_var_off,LEARNING_RATE_OFF)

    label_off_pre = tf.argmax(pre_offset,1)

    tf.summary.histogram('pre_offset', label_off_pre)
    tf.summary.histogram('offsets', offsets)
    # tf.summary.scalar('correct_prediction', tf.cast(tf.reduce_mean(correct_prediction),tf.int32))
    # tf.summary.histogram('b1', b1)

    # Define the summary writer
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(LOGS_DIR, 'train'))

    # Initialize the two dataset.
    print("setting up dataset...")
    dataset = TrainDataset(DATA_DIR)
    validation_dataset = TestDataset(DATA_DIR)

    # Define tensorflow session config.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Initialize variables.
    sess.run(tf.global_variables_initializer())

    # Restore model.
    print("Setting up Saver...")
    saver = tf.train.Saver()
    step = 0  # start_step
    ckpt = tf.train.latest_checkpoint(LOGS_DIR)
    if ckpt:
        saver.restore(sess, ckpt)
        print("Model restored...")
        print("restore from the checkpoint {0}".format(ckpt))
        print(ckpt.split('-'))
        step += int(ckpt.split('-')[-1])
    print(step)

    total_parameters = 0
    for variable in tf.trainable_variables():
        variable_parameters = 1
        for dim in variable.get_shape():
            variable_parameters *= dim.value
        total_parameters += variable_parameters

    print("Total number of trainable parameters: %d" % total_parameters)
    while step < MAX_STEPS:
        # Train the loss.
        file_v = open('./measure/train_off_acc.txt', 'a')
        start_time = time.time()
        step = step + 1
        train_images, train_annotations, train_offset = dataset.get_batch(BATCH_SIZE)
        train_offset = np.reshape(train_offset, [BATCH_SIZE, 1])
        feed_dict = {images: train_images, annotations: train_annotations, offsets:train_offset,keep_probability: 0.2}

        # Calculate training loss and print.
        if step%2 != 0:
            _, train_loss, summary,loss_off,pre_off,label_off = sess.run([train_op_seg, loss, merged,loss_offsets,pre_offset,label_off_pre], feed_dict=feed_dict)
        else:
            _, train_loss, summary, loss_off, pre_off, label_off = sess.run(
                [train_op_off, loss, merged, loss_offsets, pre_offset, label_off_pre], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        end_times = time.time()
        use_time = end_times - start_time
        print("Step: %d, Train_loss:%g, loss_off:%g,selected number:%d, step time:%g" % (step, train_loss, loss_off,dataset.cur_index, use_time))
        count = 0
        for i in range(BATCH_SIZE):
            if label_off[i]==train_offset[i]:
                count = count + 1
        print("count",count)

        file_v.writelines(str(count/BATCH_SIZE))
        file_v.writelines("\n")
        file_v.close()
        if step % 1000 == 0:
            print("--------------saving model %d----------------------" % step)
            saver.save(sess, os.path.join(LOGS_DIR, "model.ckpt"), step)

    train_writer.close()

if __name__ == '__main__':
        tf.app.run()
'''
        if step % 1000 == 0:
            metric = {
                "mse": [],
                "recall": [],
                "precision": [],
                "f_measure": [],
                "accuracy": [],
                "valid_loss": [],
                "p_acc":[],
                "m_acc":[],
                "wiou":[],
                "miou":[],
                "seg_loss": [],
                "off_loss": [],
                "off_acc":[],
            }
            file_mse = open('./measure/mse.txt', 'a')
            file_recall = open('./measure/recall.txt', 'a')
            file_precision = open('./measure/precision.txt', 'a')
            file_fmeasure = open('./measure/f_measure.txt', 'a')
            file_accuracy = open('./measure/accuracy.txt', 'a')
            file_valid_loss = open('./measure/valid_loss.txt', 'a')
            file_p_acc = open('./measure/p_acc.txt', 'a')
            file_m_acc = open('./measure/m_acc.txt', 'a')
            file_miou = open('./measure/miou.txt', 'a')
            file_wiou = open('./measure/wiou.txt', 'a')
            file_seg_loss = open('./measure/valid_seg_loss.txt', 'a')
            file_off_loss = open('./measure/valid_off_loss.txt', 'a')
            file_off_acc = open('./measure/off_acc.txt', 'a')
            valid_count = 0
            for itr in range(0,1):
                valid_image, valid_annotations, valid_offset = validation_dataset.get_batch(100)
                valid_offset = np.reshape(valid_offset, [100, 1])
                feed_dict = {images: valid_image, annotations: valid_annotations, offsets:valid_offset, keep_probability: 1.0}
                valid_loss,valid_seg_loss,valid_off_loss, pred, valid_pre_off,valid_label_off = sess.run([loss,seg_loss4 , loss_offsets,pred_annotation,pre_offset,label_off_pre], feed_dict=feed_dict)
                # print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                for i in range(100):
                    if valid_label_off[i] == valid_offset[i][0]:
                        valid_count = valid_count + 1
                    ann = np.squeeze(valid_annotations[i], axis=2)
                    image = valid_image[i]
                    pred_img = np.squeeze(pred[i], axis=2)
                    mse, recall, precision, f_measure, accuracy = evaluating_indicator(pred_img, ann, image, i)
                    pixel_acc, mean_acc, miou, weighted_iu = metrics(ann, pred_img, NUM_OF_CLASSESS)

                    if math.isnan(mse):
                        mse = np.nan_to_num(mse)
                        metric["mse"].append(mse)
                    elif math.isnan(recall):
                        recall = np.nan_to_num(recall)
                        metric["recall"].append(recall)
                    elif math.isnan(precision):
                        precision = np.nan_to_num(precision)
                        metric["precision"].append(precision)
                    elif math.isnan(f_measure):
                        f_measure = np.nan_to_num(f_measure)
                        metric["f_measure"].append(f_measure)
                    elif math.isnan(accuracy):
                        accuracy = np.nan_to_num(accuracy)
                        metric["accuracy"].append(accuracy)
                    elif math.isnan(valid_loss):
                        valid_loss = np.nan_to_num(valid_loss)
                        metric["valid_loss"].append(valid_loss)
                    elif math.isnan(valid_seg_loss):
                        valid_seg_loss = np.nan_to_num(valid_seg_loss)
                        metric["seg_loss"].append(valid_seg_loss)
                    elif math.isnan(valid_off_loss):
                        valid_off_loss = np.nan_to_num(valid_off_loss)
                        metric["off_loss"].append(valid_off_loss)
                    elif math.isnan(pixel_acc):
                        pixel_acc = np.nan_to_num(pixel_acc)
                        metric["p_acc"].append(pixel_acc)
                    elif math.isnan(mean_acc):
                        mean_acc = np.nan_to_num(mean_acc)
                        metric["m_acc"].append(mean_acc)
                    elif math.isnan(miou):
                        miou = np.nan_to_num(miou)
                        metric["miou"].append(miou)
                    elif math.isnan(weighted_iu):
                        weighted_iu = np.nan_to_num(weighted_iu)
                        metric["wiou"].append(weighted_iu)
                    else:
                        metric["mse"].append(mse)
                        metric["recall"].append(recall)
                        metric["precision"].append(precision)
                        metric["f_measure"].append(f_measure)
                        metric["accuracy"].append(accuracy)
                        metric["valid_loss"].append(valid_loss)
                        metric["p_acc"].append(pixel_acc)
                        metric["m_acc"].append(mean_acc)
                        metric["miou"].append(miou)
                        metric["wiou"].append(weighted_iu)
                        metric["off_loss"].append(valid_off_loss)
                        metric["seg_loss"].append(valid_seg_loss)

            metric["off_acc"].append(valid_count/100)
            for key in metric:
                metric[key] = np.mean(metric[key])
                print("%s:" % key, metric[key])

            file_mse.writelines(str(np.mean(metric["mse"])))
            file_mse.writelines("\n")
            file_mse.close()
            file_recall.writelines(str(np.mean(metric["recall"])))
            file_recall.writelines("\n")
            file_recall.close()
            file_precision.writelines(str(np.mean(metric["precision"])))
            file_precision.writelines("\n")
            file_precision.close()
            file_fmeasure.writelines(str(np.mean(metric["f_measure"])))
            file_fmeasure.writelines("\n")
            file_fmeasure.close()
            file_accuracy.writelines(str(np.mean(metric["accuracy"])))
            file_accuracy.writelines("\n")
            file_accuracy.close()
            file_valid_loss.writelines(str(np.mean(metric["valid_loss"])))
            file_valid_loss.writelines("\n")
            file_valid_loss.close()
            file_p_acc.writelines(str(np.mean(metric["p_acc"])))
            file_p_acc.writelines("\n")
            file_p_acc.close()
            file_m_acc.writelines(str(np.mean(metric["m_acc"])))
            file_m_acc.writelines("\n")
            file_m_acc.close()
            file_miou.writelines(str(np.mean(metric["miou"])))
            file_miou.writelines("\n")
            file_miou.close()
            file_wiou.writelines(str(np.mean(metric["wiou"])))
            file_wiou.writelines("\n")
            file_wiou.close()
            file_seg_loss.writelines(str(np.mean(metric["seg_loss"])))
            file_seg_loss.writelines("\n")
            file_seg_loss.close()
            file_off_loss.writelines(str(np.mean(metric["off_loss"])))
            file_off_loss.writelines("\n")
            file_off_loss.close()
            file_off_acc.writelines(str(metric["off_acc"]))
            file_off_acc.writelines("\n")
            file_off_acc.close()
'''




'''
        if step % 50 == 0:
            file_p_acc = open('./measure/p_acc.txt', 'a')
            file_m_acc = open('./measure/m_acc.txt', 'a')
            file_miou = open('./measure/miou.txt', 'a')
            file_wiou = open('./measure/wiou.txt', 'a')
            metric = {
                "p_acc": [],
                "m_acc": [],
                "miou": [],
                "wiou": []
            }
            pred_save = []
            image_save = []
            ann_save = []
            for itr in range(0, 100):
                # Calculate validation loss and print.
                valid_image, valid_annotations = validation_dataset.next_image()
                feed_dict = {images: valid_image, annotations: valid_annotations, keep_probability: 1.0}
                valid_loss, pred = sess.run([loss, pred_annotation], feed_dict=feed_dict)
                # print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                # pred = np.squeeze(pred, axis=3)
                pred_save = pred.copy()
                image_save = valid_image.copy()
                pred = pred[0, :, :]
                valid_annotations = np.squeeze(valid_annotations[0], axis=2)
                ann_save = valid_annotations.copy()

                # Evaluate metrics.
                pixel_acc, mean_acc, miou, weighted_iu = metrics(valid_annotations, pred, NUM_OF_CLASSESS)
                if math.isnan(pixel_acc):
                    pixel_acc = np.nan_to_num(pixel_acc)
                elif math.isnan(mean_acc):
                    mean_acc = np.nan_to_num(mean_acc)
                elif math.isnan(miou):
                    miou = np.nan_to_num(miou)
                elif math.isnan(weighted_iu):
                    weighted_iu = np.nan_to_num(weighted_iu)
                else:
                    metric["p_acc"].append(valid_loss)
                    metric["m_acc"].append(mean_acc)
                    metric["miou"].append(miou)
                    metric["wiou"].append(weighted_iu)

                    # Print result.
                print("picture %d" % validation_dataset.cur_index, pixel_acc, mean_acc, miou, weighted_iu)

            for key in metric:
                metric[key] = np.mean(metric[key])
                print("%s:" % key, metric[key])
            file_p_acc.writelines(str(np.mean(metric["p_acc"])))
            file_p_acc.writelines("\n")
            file_p_acc.close()
            file_m_acc.writelines(str(np.mean(metric["m_acc"])))
            file_m_acc.writelines("\n")
            file_m_acc.close()
            file_miou.writelines(str(np.mean(metric["miou"])))
            file_miou.writelines("\n")
            file_miou.close()
            file_wiou.writelines(str(np.mean(metric["wiou"])))
            file_wiou.writelines("\n")
            file_wiou.close()

            if metric["wiou"] > 0.3:
                output_path = os.path.join(OUTPUT_DIR+'/val', "valid_" + str(step) + ".png")
                image_path = os.path.join(OUTPUT_DIR+'/img', "image_" + str(step) + ".png")
                ann_path = os.path.join(OUTPUT_DIR+'/gt', "ann_" + str(step) + ".png")
                pred_save = np.squeeze(pred_save[0], axis=2)
                image = np.squeeze(image_save, axis=0)
                io.imsave(image_path, image)
                io.imsave(output_path, pred_save.astype(np.uint8) * 100)
                io.imsave(ann_path, ann_save)
                print("Saved image: %s" % output_path)

            # Save the model.
'''

