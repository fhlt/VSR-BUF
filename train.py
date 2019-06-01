import tensorflow as tf
import numpy as np
from model import build_BUF, freeze_graph
from data_utils import TrainsetLoader

T_in = 7  # Size of input temporal radius
R = 4  # Upscaling factor
n_iters = 3000  # total train epochs
learning_rate = 1e-4 # learning rate
trainset_dir = "data/train/"
upscale_factor = 4
batch_size = 1


H_out_true=tf.placeholder(tf.float32,shape=(None, None, None, 3),name='H_out_true') 
is_train = tf.placeholder(tf.bool, shape=[],name='is_train')  # Phase ,scalar
L =  tf.placeholder(tf.float32, shape=[None, T_in, None, None, 3],name='L_in')
train_loader = TrainsetLoader(trainset_dir, upscale_factor, batch_size, n_iters, T_in)
# build model
cost, learning_rate_decay_op, optimizer = build_BUF(H_out_true, is_train, L, learning_rate)
# Session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for idx_iter in range(len(train_loader)):
        x_train_batch, y_train_batch = train_loader.__getitem__(idx_iter)
        train_loss, _ = sess.run([cost, optimizer], feed_dict={H_out_true:y_train_batch, L:x_train_batch, is_train:True})
        print("train cost :" + train_loss)
        if idx_iter==0:
            with open('./logs/pb_graph_log.txt', 'w') as f:
                f.write(str(sess.graph_def)) 
            var_list = tf.global_variables()
            with open('./logs/global_variables_log.txt','w') as f:
                f.write(str(var_list)) 
        '''
        tf.train.write_graph(sess.graph_def, '.', './checkpoint/duf_'+str(global_step)+'.pbtxt')
        saver.save(sess, save_path="./checkpoint/duf",global_step=global_step)
        freeze_graph(check_point_folder='./checkpoint/',model_folder='./model',pb_name='My_Duf_'+str(global_step)+'.pb')
        '''
