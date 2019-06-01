import tensorflow as tf
from utils import BatchNorm,Conv3D,DynFilter3D,depth_to_space_3D,Huber, LoadImage
import numpy as np
import glob
from tensorflow.python.framework import graph_util

T_in = 7  # Size of input temporal radius
R = 4  # Upscaling factor

def freeze_graph(check_point_folder,model_folder,pb_name):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(check_point_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    output_graph=model_folder+'/'+pb_name
    # Before exporting our graph, we need to precise what is our output node
    # this variables is plural, because you can have multiple output nodes
    output_node_names = "out_H"
    list_str =[]

    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    clear_devices = True

    # We import the meta graph and retrive a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        # fix batch norm nodes
        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")  # We split on comma for convenience
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

# Network
def build_BUF(H_out_true, is_train, L, learning_rate):
    # build model
    stp = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
    sp = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]
    # [1, 3, 3, 3, 64] [filter_depth, filter_height, filter_width, in_channels,out_channels]
    x = Conv3D(tf.pad(L, sp, mode='CONSTANT'), [1, 3, 3, 3, 64], [1, 1, 1, 1, 1], 'VALID', name='conv1')

    F = 64
    G = 32
    for r in range(3): 
        t = BatchNorm(x, is_train, name='Rbn' + str(r + 1) + 'a')
        t = tf.nn.relu(t)
        t = Conv3D(t, [1, 1, 1, F, F], [1, 1, 1, 1, 1], 'VALID', name='Rconv' + str(r + 1) + 'a')

        t = BatchNorm(t, is_train, name='Rbn' + str(r + 1) + 'b')
        t = tf.nn.relu(t)
        t = Conv3D(tf.pad(t, stp, mode='CONSTANT'), [3, 3, 3, F, G], [1, 1, 1, 1, 1], 'VALID',
                name='Rconv' + str(r + 1) + 'b')

        x = tf.concat([x, t], 4)
        F += G
    for r in range(3, 6):
        t = BatchNorm(x, is_train, name='Rbn' + str(r + 1) + 'a')
        t = tf.nn.relu(t)
        t = Conv3D(t, [1, 1, 1, F, F], [1, 1, 1, 1, 1], 'VALID', name='Rconv' + str(r + 1) + 'a')

        t = BatchNorm(t, is_train, name='Rbn' + str(r + 1) + 'b')
        t = tf.nn.relu(t)
        t = Conv3D(tf.pad(t, sp, mode='CONSTANT'), [3, 3, 3, F, G], [1, 1, 1, 1, 1], 'VALID',
                name='Rconv' + str(r + 1) + 'b')

        x = tf.concat([x[:, 1:-1], t], 4)
        F += G

    # sharen section
    x = BatchNorm(x, is_train, name='fbn1')
    x = tf.nn.relu(x)
    x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1, 3, 3, 256, 256], [1, 1, 1, 1, 1], 'VALID', name='conv2')
    x = tf.nn.relu(x)

    # R
    r = Conv3D(x, [1, 1, 1, 256, 256], [1, 1, 1, 1, 1], 'VALID', name='rconv1')
    r = tf.nn.relu(r)
    r = Conv3D(r, [1, 1, 1, 256, 3 * 16], [1, 1, 1, 1, 1], 'VALID', name='rconv2')

    # F
    f = Conv3D(x, [1, 1, 1, 256, 512], [1, 1, 1, 1, 1], 'VALID', name='fconv1')
    f = tf.nn.relu(f)
    f = Conv3D(f, [1, 1, 1, 512, 1 * 5 * 5 * 16], [1, 1, 1, 1, 1], 'VALID', name='fconv2')

    ds_f = tf.shape(f)
    f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], ds_f[3], 25, 16])
    f = tf.nn.softmax(f, dim=4)

    Fx=f
    Rx =r

    x=L
    x_c = []
    for c in range(3):
        t = DynFilter3D(x[:, T_in // 2:T_in // 2 + 1, :, :, c], Fx[:, 0, :, :, :, :], [1, 5, 5])  # [B,H,W,R*R]
        t = tf.depth_to_space(t, R)  # [B,H*R,W*R,1]
        x_c += [t]
    x = tf.concat(x_c, axis=3)  # [B,H*R,W*R,3] Tensor("concat_9:0", shape=(?, ?, ?, 3), dtype=float32)

    x = tf.expand_dims(x, axis=1) # Tensor("ExpandDims_3:0", shape=(?, 1, ?, ?, 3), dtype=float32)
    Rx = depth_to_space_3D(Rx, R)  # [B,1,H*R,W*R,3] Tensor("Reshape_6:0", shape=(?, ?, ?, ?, ?), dtype=float32)
    x += Rx # Tensor("add_18:0", shape=(?, ?, ?, ?, 3), dtype=float32) 
    x = tf.squeeze(x)
    print(x.get_shape())
    out_H = tf.clip_by_value(x,0,1,name='out_H')
    cost = Huber(y_true=H_out_true,y_pred=out_H,delta=0.01)
    learning_rate = learning_rate
    learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32,name='learning_rate')
    learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    return cost, learning_rate_decay_op, optimizer