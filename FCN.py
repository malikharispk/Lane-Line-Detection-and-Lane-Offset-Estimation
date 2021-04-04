import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import FCN_down_sizing
from tensorflow.contrib.layers.python.layers import utils as tf_utils


def conv3x3(inputs, filters, rate, name):
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=[3, 3],
                            padding="same",
                            activation=tf.nn.relu,
                            dilation_rate=rate,
                            name=name)


def conv1(inputs,channel,rate,scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        conv1x1 = slim.conv2d(inputs, channel, [1, 1], stride=1, rate=1, scope='_conv1x1')
        conv1x1 = slim.batch_norm(conv1x1, scope='_bn1x1')
        conv1x1 = tf.nn.relu(conv1x1)

    return conv1x1


def deconv(inputs, filters, kernel_size, stride=2):
    return tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides=[stride, stride], padding="same")


def upsampling2d(input_, size=(2, 2), name='upsampling2d'):
    #use:up1 = tf_utils.upsampling2d(deconv3, size=(2, 2), name='up1')
    with tf.name_scope(name):
        shape = input_.get_shape().as_list()
        return tf.image.resize_nearest_neighbor(input_, size=(size[0] * shape[1], size[1] * shape[2]))



def max_pool_2x2(inputs):
    return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(inputs):
    return tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def __root_block(inputs, scope=None):
    # with tf.variable_scope(scope, 'root', [inputs]) as sc:
    with tf.variable_scope(scope, 'root') as sc:
        with slim.arg_scope([slim.conv2d], padding='SAME',
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            # weights_initializer = tf.contrib.layers.xavier_initializer(),
            weights_regularizer = slim.l2_regularizer(0.0005),
            activation_fn=None):
            net = slim.conv2d(inputs, 64, [3, 3], stride=2, scope='conv1')
            net = slim.batch_norm(net, scope='bn1')
            net = tf.nn.relu(net)
            net = slim.conv2d(net, 64, [3, 3], stride=1, scope='conv2')
            net = slim.batch_norm(net, scope='bn2')
            net = tf.nn.relu(net)
            net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv3')
            net = slim.batch_norm(net, scope='bn3')
            net = tf.nn.relu(net)
            # print(net.get_shape())
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', padding='SAME')
    return net

def __root_block_2(inputs,keep_prob, scope=None):
    # with tf.variable_scope(scope, 'root', [inputs]) as sc:
    with tf.variable_scope(scope, 'root') as sc:
        with slim.arg_scope([slim.conv2d], padding='SAME',
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            # weights_initializer = tf.contrib.layers.xavier_initializer(),
            weights_regularizer = slim.l2_regularizer(0.0005),
            activation_fn=None):
            conv0 = slim.conv2d(inputs, 16, [7, 7], stride=1, rate=9,scope='conv0')
            conv0 = slim.batch_norm(conv0, scope='bn0')
            conv0 = tf.nn.relu(conv0)
            conv1 = slim.conv2d(conv0, 16, [3, 3], stride=1, rate=7,scope='conv1_1')
            conv1 = slim.batch_norm(conv1, scope='bn1_1')
            conv1 = tf.nn.relu(conv1)
            conv1 = slim.conv2d(conv1, 16, [3, 3], stride=1, rate=5,scope='conv1_2')
            conv1 = slim.batch_norm(conv1, scope='bn1_2')
            conv1 = tf.nn.relu(conv1)
            conv1 = slim.conv2d(conv1, 16, [3, 3], stride=1, rate=3,scope='conv1_3')
            conv1 = slim.batch_norm(conv1, scope='bn1_3')
            conv1 = tf.nn.relu(conv1)
            split1 = tf.split(conv1, num_or_size_splits=16, axis=3)
            tf.summary.image('conv1', split1[0]*200, 2)
            # conv1 = slim.conv2d(conv1, 32, [3, 3], stride=1, rate=9,scope='conv1_2')
            # conv1 = slim.batch_norm(conv1, scope='bn1_2')
            # conv1 = tf.nn.relu(conv1)
            conv1 = slim.max_pool2d(conv1, [3, 3], stride=2, scope='pool1', padding='SAME')

            conv2 = slim.conv2d(conv1, 32, [3, 3], stride=1, rate=5,scope='conv2_1')
            conv2 = slim.batch_norm(conv2, scope='bn2_1')
            conv2 = tf.nn.relu(conv2)
            conv2 = slim.conv2d(conv2, 32, [3, 3], stride=1, rate=3,scope='conv2_2')
            conv2 = slim.batch_norm(conv2, scope='bn2_2')
            conv2 = tf.nn.relu(conv2)
            conv2 = slim.conv2d(conv2, 32, [3, 3], stride=1, rate=2,scope='conv2_3')
            conv2 = slim.batch_norm(conv2, scope='bn2_3')
            conv2 = tf.nn.relu(conv2)
            conv2 = slim.conv2d(conv2, 32, [3, 3], stride=1, rate=1,scope='conv2_4')
            conv2 = slim.batch_norm(conv2, scope='bn2_4')
            conv2 = tf.nn.relu(conv2)
            split2 = tf.split(conv2, num_or_size_splits=32, axis=3)
            tf.summary.image('conv2', split2[0]*200, 2)
            conv2 = slim.max_pool2d(conv2, [3, 3], stride=2, scope='pool2', padding='SAME')

            conv3 = slim.conv2d(conv2, 64, [3, 3], stride=1, rate=1, scope='conv3_1')
            conv3 = slim.batch_norm(conv3, scope='bn3_1')
            conv3 = tf.nn.relu(conv3)
            conv3 = slim.conv2d(conv3, 64, [3, 3], stride=1, rate=1, scope='conv3_2')
            conv3 = slim.batch_norm(conv3, scope='bn3_2')
            conv3 = tf.nn.relu(conv3)
            conv3 = slim.conv2d(conv3, 64, [3, 3], stride=1, rate=1, scope='conv3_3')
            conv3 = slim.batch_norm(conv3, scope='bn3_3')
            conv3 = tf.nn.relu(conv3)
            conv3 = slim.conv2d(conv3, 64, [3, 3], stride=1, rate=1, scope='conv3_4')
            conv3 = slim.batch_norm(conv3, scope='bn3_4')
            conv3 = tf.nn.relu(conv3)
            # conv3 = tf.nn.dropout(conv3, keep_prob=keep_prob)
            # split3 = tf.split(conv3, num_or_size_splits=64, axis=3)
            # tf.summary.image('conv3', split3[0], 2)
            conv3 = slim.max_pool2d(conv3, [3, 3], stride=2, scope='pool3', padding='SAME')

    return conv3, conv2, conv1

def __root_block_resnet(inputs, scope=None):
    net = slim.conv2d(inputs, 16, [5, 5], stride=1)
    net = slim.batch_norm(net, scope='net')
    net = tf.nn.relu(net)
    res_1 = __residual_block(net, 32, 2, 1)
    res_2 = __residual_block(res_1, 32, 1, 7)
    res_3 = __residual_block(res_2, 64, 2, 1)
    res_4 = __residual_block(res_3, 64, 1, 5)
    res_5 = __residual_block(res_4, 128, 2, 1)
    res_6 = __residual_block(res_5, 128, 1, 2)
    return res_6,res_4,res_2

def __residual_block(inputs, output_num, stride=1, dilate_rate=1, scope=None):
    # with tf.variable_scope(scope, 'residual_block', [inputs]) as sc:
    with tf.variable_scope(scope, 'residual_block') as sc:
        with slim.arg_scope([slim.conv2d], padding='SAME',
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            # weights_initializer = tf.contrib.layers.xavier_initializer(),
            weights_regularizer = slim.l2_regularizer(0.0005),
            activation_fn=None):

            depth_in = tf_utils.last_dimension(inputs.get_shape(), min_rank=4)

            depth_bottleneck = depth_in // 4

            if output_num == depth_in and stride == 1:
                shortcut = inputs
            else:
                shortcut = slim.conv2d(inputs, output_num, [1, 1], stride=stride,  scope='shortcut-conv')
                shortcut = slim.batch_norm(shortcut, scope='shortcut-bn')

            residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=stride, scope='conv1')
            residual = slim.batch_norm(residual, scope='bn1')
            residual = tf.nn.relu(residual)

            residual = slim.conv2d(residual, depth_bottleneck, [3, 3], stride=1, rate=dilate_rate, scope='conv2')
            residual = slim.batch_norm(residual, scope='bn2')
            residual = tf.nn.relu(residual)

            residual = slim.conv2d(residual, output_num, [1, 1], stride=1, scope='conv3')
            residual = slim.batch_norm(residual, scope='bn3')

            output = tf.nn.relu(shortcut + residual)

    return output


def __pyramid_pooling_module(inputs, scope = None):
    def branch(inputs, bin_size, rate, name):
        inputs_shape = inputs.get_shape()
        pool_size = bin_size
        print('name: %s, shape: %d, bin_size: %d' % (name, inputs_shape[1], bin_size))
        # with tf.variable_scope(scope, 'branch_block_%s' % name, [inputs]) as sc:
        with tf.variable_scope(scope, 'branch_block_%s' % name) as sc:
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                # weights_initializer = tf.contrib.layers.xavier_initializer(),
                weights_regularizer=slim.l2_regularizer(0.0005),
                activation_fn=None):

                dims = inputs.get_shape().dims
                out_height, out_width, depth = dims[1].value, dims[2].value, dims[3].value


                conv1 = slim.conv2d(inputs, depth, [3, 3], stride=1, scope='conv1',rate = rate)
                pool1 = slim.avg_pool2d(conv1, pool_size, stride=pool_size, padding='SAME', scope='pool1')
                bn1 = slim.batch_norm(pool1, scope='bn1')
                relu1 = tf.nn.relu(bn1, name='relu1')

                # output = tf.image.resize_bilinear(relu1, [out_height, out_width])

                output = slim.conv2d_transpose(relu1, depth, [3, 3], [pool_size, pool_size])

        return output
        # with tf.variable_scope(scope, 'pyramid_pooling_module', [inputs]) as sc:

    with tf.variable_scope(scope, 'pyramid_pooling_module') as sc:
        branchs = [inputs]
        for [bin,rate] in [[1,7], [2,3], [4,1], [2,3],[1,7]]:
            b = branch(inputs, bin_size=bin, rate=rate, name='branch_bin_%d' % bin)
            print('bin',bin,rate)
            branchs.append(b)
        net = tf.concat(axis=3, values=branchs)
        pass

    return net

def pspnet(inputs, keep_prob, num_classes, reuse=False):
    # with tf.variable_scope(scope, 'pspnet_v1', [inputs], reuse=reuse) as sc:
    with tf.variable_scope('pspnet_v1', reuse=reuse) as sc:

        end_points_collection = sc.name + '_end_points'
        # with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = __root_block(inputs)
        net_56 = net
        params = [
            {'output_num': 128, 'stride': 1, 'dilate_rate': 1},
            {'output_num': 256, 'stride': 2, 'dilate_rate': 1},
            {'output_num': 256, 'stride': 1, 'dilate_rate': 1},
            {'output_num': 512, 'stride': 2, 'dilate_rate': 1},
            {'output_num': 512, 'stride': 1, 'dilate_rate': 2},
            {'output_num': 512, 'stride': 1, 'dilate_rate': 4},
        ]
        # params = [
        #     {'output_num': 128, 'stride': 1, 'dilate_rate': 1},
        #     {'output_num': 256, 'stride': 2, 'dilate_rate': 1},
        #     {'output_num': 512, 'stride': 2, 'dilate_rate': 1},
        # ]
        for p in params:
            output_num = p['output_num']
            stride = p['stride']
            dilate_rate = p['dilate_rate']

            net = __residual_block(net, output_num, stride, dilate_rate)
        pass
        net_14 = net
        print('net_14', net_14)
        net = __pyramid_pooling_module(net)
        print(net.get_shape())
        net = tf.nn.dropout(net, keep_prob=keep_prob)

        net = slim.conv2d(net, 4096, [1, 1], stride=1, scope='fc1')
        net = tf.nn.dropout(net, keep_prob=keep_prob)

        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='logits')

        net = deconv(net, 512, kernel_size=[3, 3], stride=1)
        net = tf.add(net, net_14)

        net = deconv(net, 128, kernel_size=[8, 8], stride=4)
        # print("net_56",net_56.get_shape())
        net = tf.add(net, net_56)

        logits = deconv(net, num_classes, kernel_size=[16, 16], stride=4)  # image shape is  224 * 224
        # dims = inputs.get_shape().dims
        # out_height, out_width = dims[1].value, dims[2].value
        # logits = tf.image.resize_bilinear(net, [out_height, out_width])
        # Predicted annotation without channel dimension.
        annotation_pred = tf.argmax(logits, axis=3, name="prediction")

        # Predicted annotation with full dimension.
        expanded_annotation_pred = tf.expand_dims(annotation_pred, dim=3)
        print(logits.get_shape())

    return expanded_annotation_pred, logits

def pspnet_2(inputs, keep_prob, num_classes, reuse=False):
    # with tf.variable_scope(scope, 'pspnet_v1', [inputs], reuse=reuse) as sc:
    with tf.variable_scope('pspnet_v1', reuse=reuse) as sc:

        end_points_collection = sc.name + '_end_points'
        # with slim.arg_scope([slim.batch_norm], is_training=is_training):
        # net6, net4, net2 = __root_block_resnet(inputs)
        net6, net4, net2 = __root_block_2(inputs)
        print('net_6', net6.get_shape())
        print('net_4', net4.get_shape())
        print('net_2', net2.get_shape())
        net = __pyramid_pooling_module(net6)

        net = tf.nn.dropout(net, keep_prob=keep_prob)

        net = slim.conv2d(net, 4096, [1, 1], stride=1, scope='fc1')
        net = tf.nn.dropout(net, keep_prob=keep_prob)

        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='logits')

        net = deconv(net, 32, kernel_size=[3, 3], stride=1)
        print('net', net.get_shape())
        net = tf.add(net, net6)

        net = deconv(net, 16, kernel_size=[8, 8], stride=2)
        # print("net_56",net_56.get_shape())
        net = tf.add(net, net4)

        net = deconv(net, 8, kernel_size=[16, 16], stride=2)
        # print("net_56",net_56.get_shape())
        net = tf.add(net, net2)


        logits = deconv(net, num_classes, kernel_size=[32, 32], stride=2)
        # dims = inputs.get_shape().dims
        # out_height, out_width = dims[1].value, dims[2].value
        # logits = tf.image.resize_bilinear(net, [out_height, out_width])
        # Predicted annotation without channel dimension.
        annotation_pred = tf.argmax(logits, axis=3, name="prediction")

        # Predicted annotation with full dimension.
        expanded_annotation_pred = tf.expand_dims(annotation_pred, dim=3)
        print(logits.get_shape())

    return expanded_annotation_pred, logits

def pspnet_res(inputs, keep_prob, num_classes,reuse=False):
    # with tf.variable_scope(scope, 'pspnet_v1', [inputs], reuse=reuse) as sc:
    with tf.variable_scope('pspnet_v1', reuse=reuse) as sc:
        net6, net4, net2 = __root_block_2(inputs, keep_prob)

        net = __pyramid_pooling_module(net6)
        net = tf.nn.dropout(net, keep_prob=keep_prob)
        net_off = net

        # net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5', padding='SAME')
        # net = slim.avg_pool2d(net, [3, 3], stride=1, scope='pool5a', padding='SAME')

        net = slim.conv2d(net, 1024, [1, 1], stride=1, scope='fc1')
        net = slim.batch_norm(net, scope='fc_bn1')
        net = tf.nn.relu(net)
        net = tf.nn.dropout(net, keep_prob=keep_prob)

        net = slim.conv2d(net, num_classes, [1, 1], scope='logits')
        net = slim.batch_norm(net, scope='logits_bn1')
        net = tf.nn.relu(net)

        # net = tf.layers.dense(inputs=net, units=64, activation=None)
        # net = tf.nn.dropout(net, keep_prob=keep_prob)
        net = slim.batch_norm(net, scope='t_1_1')
        net = deconv(net, 64, kernel_size=[4, 4], stride=1)
        # net = tf.layers.dense(inputs=net, units=64, activation=None)
        # net = slim.batch_norm(net, scope='t_1_2')

        net_dims = net.get_shape().dims
        net_height, net_width = net_dims[1].value, net_dims[2].value
        logits1 = tf.image.resize_bilinear(net, [net_height * 8, net_width * 8])
        logits1 = slim.conv2d(logits1, num_classes, [1, 1], scope='logits1')
        split_logits1 = tf.split(logits1, num_or_size_splits=num_classes, axis=3)
        tf.summary.image('logits1', split_logits1[0] * 200, 2)

        net = tf.add(net, net6)

        # net = tf.layers.dense(inputs=net, units=32, activation=None)
        # net = slim.batch_norm(net, scope='t_2_1')
        net = deconv(net, 32, kernel_size=[4, 4], stride=2)
        # net = tf.layers.dense(inputs=net, units=32, activation=None)
        # net = slim.batch_norm(net, scope='t_2_2')

        net_dims = net.get_shape().dims
        net_height, net_width = net_dims[1].value, net_dims[2].value
        logits2 = tf.image.resize_bilinear(net, [net_height * 4, net_width * 4])
        logits2 = slim.conv2d(logits2, num_classes, [1, 1], scope='logits2')

        split_logits2 = tf.split(logits2, num_or_size_splits=num_classes, axis=3)
        tf.summary.image('logits2', split_logits2[0] * 200, 2)

        # net = tf.add(net, net4)

        # net = tf.layers.dense(inputs=net, units=16, activation=None)
        # net = slim.batch_norm(net, scope='t_3_1')
        net = deconv(net, 16, kernel_size=[8, 8], stride=2)
        # net = tf.layers.dense(inputs=net, units=16, activation=None)
        # net = slim.batch_norm(net, scope='t_3_2')

        net_dims = net.get_shape().dims
        net_height, net_width = net_dims[1].value, net_dims[2].value
        logits3 = tf.image.resize_bilinear(net, [net_height * 2, net_width * 2])
        logits3 = slim.conv2d(logits3, num_classes, [1, 1], scope='logits3')
        split_logits3 = tf.split(logits3, num_or_size_splits=num_classes, axis=3)
        tf.summary.image('logits3', split_logits3[0] * 200, 2)

        net = tf.add(net, net2)

        dims = inputs.get_shape().dims
        out_height, out_width = dims[1].value, dims[2].value
        net = tf.image.resize_bilinear(net, [out_height, out_width])
        net = tf.concat(axis=3, values=[net, logits1, logits2, logits3])
        logits = slim.conv2d(net, num_classes, [3, 3], scope='logits4')
        logits = slim.batch_norm(logits, scope='logits_bn_end1')
        logits = slim.max_pool2d(logits, [3, 3], stride=1, scope='pool5', padding='SAME')
        logits = slim.conv2d(logits, num_classes, [3, 3], scope='logits5')
        logits = slim.batch_norm(logits, scope='logits_bn_end2')
        logits = deconv(logits, num_classes, kernel_size=[16, 16], stride=1)

        offset = offset_layer(net_off,keep_prob)
        print("offset", offset.get_shape())
        # dims = inputs.get_shape().dims
        # out_height, out_width = dims[1].value, dims[2].value
        # logits = tf.image.resize_bilinear(net, [out_height, out_width])
        # Predicted annotation without channel dimension.
        annotation_pred = tf.argmax(logits, axis=3, name="prediction")

        # Predicted annotation with full dimension.
        expanded_annotation_pred = tf.expand_dims(annotation_pred, dim=3)
        print(logits.get_shape())

    return expanded_annotation_pred, logits, logits3, logits2, logits1,offset

def get_fcn_32s_net(image, keep_prob, num_of_class):
    """
    Construct FCN-8s net and return the prediction layers.

    :param image: numpy.ndarray, (batch_size, height, width, channel).
        The image or annotation to be processed.
    :param keep_prob: the keeping probability of dropout layer.
    :param num_of_class: Total num of classes, including the "other" class.
        When the dataset is ADE20k, its value is 151.

    :return:
        expanded_annotation_pred: Tensor, (batch_size, height, width, 1).
        The predicted class index of each pixel, by calculating tf.argmax on {conv_t3}.

        conv_t3: Tensor, (batch_size, height, width, num_of_class).
        The predicted probability of each class on pixels.
        The last output layer that hasn't calculate tf.argmax.

    """
    # The down sizing part of FCN.
    pool3, pool4, conv8 = FCN_down_sizing.getNet(image, keep_prob, num_of_class)

    # Fuse layer 1,
    pool4_shape = pool4.get_shape()
    conv_t1 = deconv(conv8, pool4_shape[3].value, [4, 4])
    # fuse1 = tf.add(conv_t1, pool4)

    # Fuse layer 2.
    pool3_shape = pool3.get_shape()
    conv_t2 = deconv(conv_t1, pool3_shape[3].value, [4, 4])
    # fuse2 = tf.add(conv_t2, pool3)

    # Output layer.
    conv_t3 = deconv(conv_t2, num_of_class, kernel_size=[16, 16], stride=8)  # image shape is  224 * 224

    # Predicted annotation without channel dimension.
    annotation_pred = tf.argmax(conv_t3, axis=3, name="prediction")

    # Predicted annotation with full dimension.
    expanded_annotation_pred = tf.expand_dims(annotation_pred, dim=3)

    return expanded_annotation_pred, conv_t3

def deng_net(image,keep_prob):
    """
    :param image:
    :return:
    """
    conv1 = slim.conv2d(image, 16, [3, 3], stride=1, scope='conv1_1')
    conv1 = slim.batch_norm(conv1, scope='bn1_1')
    conv1 = tf.nn.relu(conv1)
    conv1 = slim.conv2d(conv1, 16, [3, 3], stride=1, scope='conv1_2')
    conv1 = slim.batch_norm(conv1, scope='bn1_2')
    conv1 = tf.nn.relu(conv1)
    conv1 = slim.max_pool2d(conv1, [3, 3], stride=2, scope='pool1', padding='SAME')

    conv2 = slim.conv2d(conv1, 32, [3, 3], stride=1, scope='conv2_1')
    conv2 = slim.batch_norm(conv2, scope='bn2_1')
    conv2 = tf.nn.relu(conv2)
    conv2 = slim.conv2d(conv2, 32, [3, 3], stride=1, scope='conv2_2')
    conv2 = slim.batch_norm(conv2, scope='bn2_2')
    conv2 = tf.nn.relu(conv2)
    conv2 = slim.max_pool2d(conv2, [3, 3], stride=2, scope='pool2', padding='SAME')

    conv3 = slim.conv2d(conv2, 64, [3, 3], stride=1, scope='conv3_1')
    conv3 = slim.batch_norm(conv3, scope='bn3_1')
    conv3 = tf.nn.relu(conv3)
    conv3 = slim.conv2d(conv3, 64, [3, 3], stride=1, scope='conv3_2')
    conv3 = slim.batch_norm(conv3, scope='bn3_2')
    conv3 = tf.nn.relu(conv3)
    conv3 = slim.max_pool2d(conv3, [3, 3], stride=2, scope='pool3', padding='SAME')

    conv_t1 = slim.conv2d(conv3, 4096, [1, 1], stride=1)
    conv_t1 = slim.batch_norm(conv_t1, scope='conv_t1')
    conv_t1 = tf.nn.relu(conv_t1)
    conv_t1 = tf.nn.dropout(conv_t1, keep_prob=keep_prob)
    conv_t1 = slim.conv2d(conv_t1, 32, [3, 3], stride=1)
    conv_t1 = slim.batch_norm(conv_t1, scope='conv_t2')
    conv_t1 = tf.nn.relu(conv_t1)
    conv_t1 = deconv(conv_t1, 32, kernel_size=[4, 4], stride=2)

    conv_t1 = tf.add(conv_t1,conv2)
    conv_t1 = slim.conv2d(conv_t1, 16, [3, 3], stride=1)
    conv_t1 = slim.batch_norm(conv_t1, scope='conv_t3')
    conv_t1 = tf.nn.relu(conv_t1)
    conv_t1 = slim.conv2d(conv_t1, 16, [3, 3], stride=1)
    conv_t1 = slim.batch_norm(conv_t1, scope='conv_t4')
    conv_t1 = tf.nn.relu(conv_t1)
    conv_t1 = deconv(conv_t1, 16, kernel_size=[8, 8], stride=2)

    # pred = conv_t1
    conv_t1 = slim.conv2d(conv_t1, 16, [3, 3], stride=1)
    conv_t1 = slim.batch_norm(conv_t1, scope='conv_t5')
    conv_t1 = tf.nn.relu(conv_t1)
    conv_t1 = slim.conv2d(conv_t1, 16, [3, 3], stride=1)
    conv_t1 = slim.batch_norm(conv_t1, scope='conv_t6')
    conv_t1 = tf.nn.relu(conv_t1)
    conv_t1 = deconv(conv_t1, 1, kernel_size=[8, 8], stride=2)



    out = conv_t1
    print('out',out.get_shape())

    # annotation_pred = tf.argmax(pred, axis=3, name="prediction")
    #
    # # Predicted annotation with full dimension.
    # expanded_annotation_pred = tf.expand_dims(annotation_pred, dim=3)
    # print(conv_t1.get_shape())

    return out, out

def deeplab_largfov(image, keep_prob, num_of_class,rate):

    net1 = conv3x3(image, 64/rate, 1, "conv1_1")
    net1 = conv3x3(net1, 64/rate, 1, "conv1_2")
    pool1 = slim.max_pool2d(net1, [3, 3], stride=2, scope='pool1', padding='SAME')

    net2 = conv3x3(pool1, 128/rate, 1, "conv2_1")
    net2 = conv3x3(net2, 128/rate, 1, "conv2_2")
    pool2 = slim.max_pool2d(net2, [3, 3], stride=2, scope='pool2', padding='SAME')

    net3 = conv3x3(pool2, 256/rate, 1, "conv3_1")
    net3 = conv3x3(net3, 256/rate, 1, "conv3_2")
    net3 = conv3x3(net3, 256/rate, 1, "conv3_3")
    pool3 = slim.max_pool2d(net3, [3, 3], stride=2, scope='pool3', padding='SAME')

    net4 = conv3x3(pool3, 512/rate, 1, "conv4_1")
    net4 = conv3x3(net4, 512/rate, 1, "conv4_2")
    net4 = conv3x3(net4, 512/rate, 1, "conv4_3")
    pool4 = slim.max_pool2d(net4, [3, 3], stride=1, scope='pool4', padding='SAME')

    net5 = conv3x3(pool4, 512/rate, 2, "conv5_1")
    net5 = conv3x3(net5, 512/rate, 2, "conv5_2")
    net5 = conv3x3(net5, 512/rate, 2, "conv5_3")
    # net5 = tf.nn.dropout(net5, keep_prob=keep_prob)
    pool5 = slim.max_pool2d(net5, [3, 3], stride=1, scope='pool5', padding='SAME')
    pool5 = slim.avg_pool2d(pool5, [3, 3], stride=1, scope='pool5a', padding='SAME')

    net6 = conv3x3(pool5, 1024, 12, "conv6_1")
    net6 = tf.nn.dropout(net6, keep_prob=keep_prob)
    net6 = conv1(net6, 1024, 1, "conv6_2")
    net6 = tf.nn.dropout(net6, keep_prob=keep_prob)
    # net6 = conv1(net6, num_of_class, 1, "conv6_3")

    print('net6',net6.get_shape())
    net6 = deconv(net6, num_of_class, kernel_size=[4, 4], stride=2)
    dims = image.get_shape().dims
    out_height, out_width, depth = dims[1].value, dims[2].value, dims[3].value
    # logits = tf.image.resize_bilinear(net6, [out_height, out_width])
    logits = tf.image.resize_images(net6, [out_height, out_width], method=tf.image.ResizeMethod.BILINEAR)

    offset_net = conv3x3(pool5, 512 , 3, "offset1")
    # offset_net = slim.max_pool2d(offset_net, [3, 3], stride=2, scope='offset_net_pool', padding='SAME')
    offset_net = conv3x3(offset_net, 256 , 5, "offset2")
    offset_net = conv3x3(offset_net, 256 , 7, "offset3")
    offset_net = slim.max_pool2d(offset_net, [3, 3], stride=2, scope='offset_net_pool1', padding='SAME')
    offset_net = conv3x3(offset_net, 128, 7, "offset4")
    offset_net = conv3x3(offset_net, 128, 5, "offset5")
    offset_net = conv3x3(offset_net, 128, 3, "offset6")
    offset_net = conv3x3(offset_net, 64, 2, "offset7")
    offset_net = conv3x3(offset_net, 64, 2, "offset8")
    offset_net = conv1(offset_net, 1024, 1, "offset_net1")
    offset_net = conv1(offset_net, 512, 1, "offset_net2")
    offset_net = slim.conv2d(offset_net, 1, [1, 1], stride=1, rate=1, scope='offset_net_conv1x1')
    offset_net = slim.batch_norm(offset_net, scope='offset_net_bn1x1')
    offset_net = tf.nn.relu(offset_net)
    # offset_net = tf.nn.leaky_relu(offset_net,alpha=0.5)
    # offset_net = tf.squeeze(offset_net,squeeze_dims=[3])
    offset = tf.reduce_sum(offset_net, [1, 2])
    # offset_dims = offset_net.get_shape().dims
    # offset_height, offset_width = offset_dims[1].value, offset_dims[2].value
    # offset_net = tf.reshape(offset_net, [-1, 1,offset_height*offset_width])
    # print('offset_net', offset_net.get_shape())
    #
    # offset = tf.matmul(offset_net,W1)+tf.matmul(offset_net,W2)+tf.matmul(offset_net,W3)+b1

    print('offset', offset.get_shape())



    annotation_pred = tf.argmax(logits, axis=3, name="prediction")

    # Predicted annotation with full dimension.
    expanded_annotation_pred = tf.expand_dims(annotation_pred, dim=3)


    return expanded_annotation_pred, logits,offset

def conv3_reg(inputs,channel,rate,scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:

        conv3x3_1 = slim.conv2d(inputs, channel, [3, 3], stride=1, rate=rate, scope='_conv3x3')
        conv3x3_1 = slim.batch_norm(conv3x3_1, scope='_bn3x3')
        conv3x3_1 = tf.nn.relu(conv3x3_1)
    return conv3x3_1

def conv5_reg(inputs,channel,rate,scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        conv3x3_1 = slim.conv2d(inputs, channel, [5, 5], stride=1, rate=rate, scope='_conv3x3')
        conv3x3_1 = slim.batch_norm(conv3x3_1, scope='_bn3x3')
        conv3x3_1 = tf.nn.relu(conv3x3_1)
    return conv3x3_1

def conv1_reg(inputs,channel,rate,scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        regularizer = slim.l2_regularizer(0.0005)
        conv1x1 = slim.conv2d(inputs, channel, [1, 1], stride=1, rate=1, scope='_conv1x1',weights_regularizer=regularizer)
        conv1x1 = slim.batch_norm(conv1x1, scope='_bn1x1')
        conv1x1 = tf.nn.relu(conv1x1)

    return conv1x1

def multiscale(inputs, channel,num_of_class, scope):
    with tf.variable_scope("multiscale" + scope, reuse=tf.AUTO_REUSE) as scope:
        channel3x3 = channel / 8

        # image_pool
        # dims = inputs.get_shape().dims
        # out_height, out_width, depth = dims[1].value, dims[2].value, dims[3].value
        # pool1 = slim.max_pool2d(inputs, [3, 3], stride=8, scope='tan_pool1', padding='SAME')
        # pool1_conv = slim.conv2d(pool1, 11, [1, 1], stride=1, scope='tan_pool1_conv', rate=1)
        # pool1_conv = slim.batch_norm(pool1_conv, scope='tan_pool1_bn')
        # pool1_conv = tf.nn.relu(pool1_conv, name='pool1_relu')
        #
        # pool2 = slim.max_pool2d(inputs, [3, 3], stride=4, scope='tan_pool2', padding='SAME')
        # pool2_conv = slim.conv2d(pool2, 12, [1, 1], stride=1, scope='tan_pool2_conv2', rate=1)
        # pool2_conv = slim.batch_norm(pool2_conv, scope='tan_pool2_bn2')
        # pool2_conv = tf.nn.relu(pool2_conv, name='pool2_relu2')
        #
        # pool1_conv = tf.image.resize_bilinear(pool1_conv, [out_height, out_width])
        # pool2_conv = tf.image.resize_bilinear(pool2_conv, [out_height, out_width])
        #
        # pool_conv = []
        # pool_conv.append(pool1_conv)
        # pool_conv.append(pool2_conv)
        # pool_conv = tf.concat(axis=3, values=pool_conv)
        # inputs
        conv_input = slim.conv2d(inputs, channel, [1, 1], stride=1, rate=1, scope='tan_conv_input')
        conv_input = slim.batch_norm(conv_input, scope='tan_bn_input')
        conv_input = tf.nn.relu(conv_input)


        # image conv 1x1
        conv1x1_1 = slim.conv2d(conv_input, channel3x3, [1, 1], stride=1, rate=1, scope='tan_conv1x1_1')
        conv1x1_1 = slim.batch_norm(conv1x1_1, scope='tan_bn1x1_1')
        conv1x1_1 = tf.nn.relu(conv1x1_1)

        # image conv 3x3 rate 1
        conv3x3_1 = slim.conv2d(conv_input, channel3x3, [3, 3], stride=1, rate=3, scope='tan_conv3x3_1')
        conv3x3_1 = slim.batch_norm(conv3x3_1, scope='tan_bn3x3_1')
        conv3x3_1 = tf.nn.relu(conv3x3_1)

        conv3x3_2 = slim.conv2d(conv_input, channel3x3, [3, 3], stride=1, rate=5, scope='tan_conv3x3_2')
        conv3x3_2 = slim.batch_norm(conv3x3_2, scope='tan_bn3x3_2')
        conv3x3_2 = tf.nn.relu(conv3x3_2)

        conv3x3_3 = slim.conv2d(conv_input, channel3x3, [3, 3], stride=1, rate=7, scope='tan_conv3x3_3')
        conv3x3_3 = slim.batch_norm(conv3x3_3, scope='tan_bn3x3_3')
        conv3x3_3 = tf.nn.relu(conv3x3_3)

        conv3x3_4 = slim.conv2d(conv_input, channel3x3, [3, 3], stride=1, rate=9, scope='tan_conv3x3_4')
        conv3x3_4 = slim.batch_norm(conv3x3_4, scope='tan_bn3x3_4')
        conv3x3_4 = tf.nn.relu(conv3x3_4)

        conv3x3_5 = slim.conv2d(conv_input, channel3x3, [3, 3], stride=1, rate=2, scope='tan_conv3x3_5')
        conv3x3_5 = slim.batch_norm(conv3x3_5, scope='tan_bn3x3_5')
        conv3x3_5 = tf.nn.relu(conv3x3_5)

        conv5x5_1 = slim.conv2d(conv_input, channel3x3, [5, 5], stride=1, rate=1, scope='tan_conv5x5_1')
        conv5x5_1 = slim.batch_norm(conv5x5_1, scope='tan_bn5x5_1')
        conv5x5_1 = tf.nn.relu(conv5x5_1)

        conv5x5_2 = slim.conv2d(conv_input, channel3x3, [5, 5], stride=1, rate=3, scope='tan_conv5x5_2')
        conv5x5_2 = slim.batch_norm(conv5x5_2, scope='tan_bn5x5_2')
        conv5x5_2 = tf.nn.relu(conv5x5_2)
        # concat channel
        output = []
        output.append(conv1x1_1)
        output.append(conv3x3_1)
        output.append(conv3x3_2)
        output.append(conv3x3_3)
        output.append(conv3x3_4)
        output.append(conv3x3_5)
        output.append(conv5x5_1)
        output.append(conv5x5_2)
        output = tf.concat(axis=3, values=output)

        conv_output = slim.conv2d(output, num_of_class, [1, 1], stride=1, rate=1, scope='tan_conv_output')
        conv_output = slim.batch_norm(conv_output, scope='tan_bn_output')
        conv_output = tf.nn.relu(conv_output)

    return conv_output

def offset_layer(inputs,keep_prob):
    offset_net = conv3_reg(inputs, 32, 1, "offset1")
    offset_net = tf.nn.dropout(offset_net, keep_prob=keep_prob)
    # offset_net = slim.avg_pool2d(offset_net, [3, 3], stride=2, scope='offset_net_pool1', padding='SAME')
    offset_net = conv3_reg(offset_net, 32, 1, "offset2")
    offset_net = conv3_reg(offset_net, 64, 1, "offset3")
    offset_net = conv3_reg(offset_net, 64, 1, "offset4")
    # offset_net = slim.avg_pool2d(offset_net, [3, 3], stride=2, scope='offset_net_pool2', padding='SAME')
    # offset_net = conv3_reg(offset_net,32, 3, "offset3")
    # offset_net = slim.avg_pool2d(offset_net, [3, 3], stride=2, scope='offset_net_pool3', padding='SAME')
    # offset_net = conv3_reg(offset_net, 64, 3, "offset2")

    offset_dims = offset_net.get_shape().dims
    offset_height, offset_width,offset_channel = offset_dims[1].value, offset_dims[2].value, offset_dims[3].value
    offset_net = tf.reshape(offset_net, [-1,offset_height*offset_width*offset_channel])
    # dense1 = tf.layers.dense(inputs=offset_net, units=2048, activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),kernel_regularizer=slim.l2_regularizer(0.0005))
    #
    dense = tf.layers.dense(inputs=offset_net, units=128, activation=None)
    dense = tf.nn.dropout(dense, keep_prob=keep_prob)
    logits = tf.layers.dense(inputs=dense, units=7, activation=None)
    print("logits",logits.get_shape())
    return logits

def tan3_net(image, keep_prob, num_of_class, rate):
    net1 = conv5_reg(image, 64 / rate, 7, "conv1_1")
    net1 = tf.nn.dropout(net1, keep_prob=keep_prob)
    net1 = conv5_reg(net1, 64 / rate, 5, "conv1_2")
    net1 = tf.nn.dropout(net1, keep_prob=keep_prob)
    pool1 = slim.max_pool2d(net1, [3, 3], stride=2, scope='pool1', padding='SAME')

    net2 = conv5_reg(pool1, 128 / rate, 5, "conv2_1")
    net2 = conv5_reg(net2, 128 / rate, 3, "conv2_2")

    pool2 = slim.max_pool2d(net2, [3, 3], stride=2, scope='pool2', padding='SAME')

    net3 = conv3_reg(pool2, 256 / rate, 5, "conv3_1")
    net3 = conv3_reg(net3, 256 / rate, 3, "conv3_2")
    net3 = conv3_reg(net3, 256 / rate, 1, "conv3_3")
    pool3 = slim.max_pool2d(net3, [3, 3], stride=2, scope='pool3', padding='SAME')

    net4 = conv3_reg(pool3, 512 / rate, 1, "conv4_1")
    net4 = conv3_reg(net4, 512 / rate, 1, "conv4_2")
    net4 = tf.nn.dropout(net4, keep_prob=keep_prob)
    net4 = conv3_reg(net4, 512 / rate, 1, "conv4_3")
    pool4 = slim.max_pool2d(net4, [3, 3], stride=2, scope='pool4', padding='SAME')

    mult_layer = multiscale(pool4, 512, 5, 'mult_layer')

    net5 = deconv(mult_layer, 64, kernel_size=[8, 8], stride=2)
    net5 = tf.nn.dropout(net5, keep_prob=keep_prob)
    net5 = tf.add(net5, pool3)

    net6 = deconv(net5, 32, kernel_size=[16, 16], stride=2)
    net6 = tf.add(net6, pool2)
    net6 = deconv(net6, 128, kernel_size=[16, 16], stride=2)
    # net6 = deconv(net6, 128, kernel_size=[24, 24], stride=2)
    net6 = tf.layers.conv2d_transpose(net6, 128, [16, 16], strides=[2, 2], padding="same")
    logits = deconv(net6, num_of_class, kernel_size=[32, 32], stride=2)
    offset = offset_layer(pool1,keep_prob)

    annotation = tf.argmax(logits, axis=3, name="prediction")

    # Predicted annotation with full dimension.
    expanded_annotation = tf.expand_dims(annotation, dim=3)

    return expanded_annotation, logits, offset


def smooth_L1_loss(self, y_true, y_pred):
    '''
    Compute smooth L1 loss, see references.

    Arguments:
        y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
            In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
            contains the ground truth bounding box coordinates, where the last dimension
            contains `(xmin, xmax, ymin, ymax)`.
        y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
            the predicted data, in this context the predicted bounding box coordinates.

    Returns:
        The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
        of shape (batch, n_boxes_total).

    References:
        https://arxiv.org/abs/1504.08083
    '''
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred)**2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)




