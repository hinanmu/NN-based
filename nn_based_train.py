#@Time      :2018/10/30 19:03
#@Author    :zhounan
# @FileName: nn_based_train.py
import numpy as np
import tensorflow as tf
import time, threading
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.externals import joblib
from operator import itemgetter

def get_args():
    ## hyperparameters
    parser = argparse.ArgumentParser(description='neural networks for multilabel learning')
    parser.add_argument('--dataset_name', type=str, default='yeast', help='train data name')
    parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
    parser.add_argument('--epoch', type=int, default=100, help='#epoch of training')
    parser.add_argument('--hidden_unit', type=int, default=500, help='#dim of hidden state')
    parser.add_argument('--optimizer', type=str, default='Adagrad', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout keep_prob')
    parser.add_argument('--use_GPU', type=bool, default=False, help='gpu or not')
    args = parser.parse_args()

    return args

def train(data_x, data_y, args):
    data_num = data_x.shape[0]
    feature_num = data_x.shape[1]
    label_num = data_y.shape[1]
    hidden_unit = args.hidden_unit
    alpha =args.lr
    batch_size = args.batch_size

    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float32, shape=[None, feature_num], name='input_x')
        y = tf.placeholder(tf.float32, shape=[None, label_num], name='input_y')

        w1 = tf.Variable(tf.random_normal([feature_num, hidden_unit], stddev=1, seed=1))
        w2 = tf.Variable(tf.random_normal([hidden_unit, label_num], stddev=1, seed=1))

        bias1 = tf.Variable(tf.random_normal([hidden_unit], stddev=0.1, seed=1))
        bias2 = tf.Variable(tf.random_normal([label_num], stddev=0.1, seed=1))

        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        a = tf.nn.relu(tf.matmul(x, w1) + bias1)
        a = tf.nn.dropout(a, keep_prob)
        pred = tf.matmul(a, w2) + bias2
        pred = tf.nn.dropout(pred, args.dropout)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y)
        pred_prob = tf.nn.softmax(pred)
        tf.add_to_collection('pred_prob_network', pred_prob)
        optimazer = tf.train.AdagradOptimizer(alpha).minimize(loss)

    gpu_options = tf.GPUOptions(allow_growth = True)
    with tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        steps = int(args.epoch * data_num / batch_size)
        print('total step:', steps)
        for i in range(steps):
            start = (i * batch_size) % data_num
            end = min(start + batch_size, data_num)

            sess.run(optimazer, feed_dict={x: data_x[start:end],
                                           y: data_y[start:end],
                                           keep_prob:args.dropout})
            if i % 1000 == 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'step:', i)

        pred_prob = sess.run(pred_prob, feed_dict={x: data_x,
                                                   keep_prob:1.0})
        train_threshold_parallel(data_x, data_y, pred_prob, alpha, args.dataset_name)
        saver = tf.train.Saver()
        saver.save(sess, './tf_model/' + args.dataset_name + '/model')
        print('tf model save path', './tf_model/' + args.dataset_name + '/model')

def get_threshold_index(i, y_i, pred_prob_i, threshold):
    label_num = y_i.shape[0]
    tup_list = []
    for j in range(len(pred_prob_i)):
        tup_list.append((pred_prob_i[j], y_i[j]))
    tup_list = sorted(tup_list, key=itemgetter(0))

    min_val = label_num
    for j in range(len(tup_list) - 1):
        val_measure = 0

        for k in range(j + 1):
            if (tup_list[k][1] == 1):
                val_measure = val_measure + 1
        for k in range(j + 1, len(tup_list)):
            if (tup_list[k][1] == 0):
                val_measure = val_measure + 1

        if val_measure < min_val:
            min_val = val_measure
            threshold[i] = (tup_list[j][0] + tup_list[j + 1][0]) / 2

#多线程计算每个标签的阈值，貌似速度并没有加快
def train_threshold_parallel(data_x, data_y, pred_prob, alpha, dataset_name):
    data_num = data_x.shape[0]
    threshold = np.zeros([data_num])

    thread_list = []
    for i in range(data_num):
        pred_prob_i = pred_prob[i, :]
        y_i = data_y[i, :]
        thread = threading.Thread(target=get_threshold_index, args=(i, y_i, pred_prob_i, threshold))
        thread_list.append(thread)
        thread.start()

        if i % 1000 == 0:
            print(i)

    for t in thread_list:
        t.join()

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print(x_train.dtype, threshold.dtype)
    linreg = Ridge(alpha=alpha)
    linreg.fit(x_train, threshold)
    joblib.dump(linreg, './sk_model/' + dataset_name + '/linear_model.pkl')
    print('sk model save path', './sk_model/' + dataset_name + '/linear_model.pkl')

#单线程计算每个标签的阈值，遇到标签数量很大的话计算很慢（因为实例多，标签排序很慢）
def train_threshold(data_x, data_y, pred_prob, alpha, dataset_name):
    print(pred_prob.shape)
    data_num = data_x.shape[0]
    label_num = data_y.shape[1]
    threshold = np.zeros([data_num])

    for i in range(data_num):
        pred_prob_i = pred_prob[i, :]
        y_i = data_y[i, :]
        tup_list = []
        for j in range(len(pred_prob_i)):
            tup_list.append((pred_prob_i[j], y_i[j]))

        tup_list = sorted(tup_list, key=itemgetter(0))
        min_val = label_num
        for j in range (len(tup_list) - 1):
            val_measure = 0

            for k in range(j + 1):
                if(tup_list[k][1] == 1):
                    val_measure = val_measure + 1
            for k in range(j + 1, len(tup_list)):
                if(tup_list[k][1] == 0):
                    val_measure = val_measure + 1

            if val_measure < min_val:
                min_val = val_measure
                threshold[i] = (tup_list[j][0] + tup_list[j + 1][0]) / 2

    print(x_train.dtype,threshold.dtype)
    linreg = Ridge(alpha=alpha)
    linreg.fit(x_train, threshold)
    joblib.dump(linreg, './sk_model/' + dataset_name + '/linear_model.pkl')

def load_data(dataset_name):
    x_train = np.load('./dataset/' + dataset_name + '/x_train.npy')
    y_train = np.load('./dataset/' + dataset_name + '/y_train.npy')
    x_test = np.load('./dataset/' + dataset_name + '/x_test.npy')
    y_test = np.load('./dataset/' + dataset_name + '/y_test.npy')

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    args = get_args()
    dataset_names = ['yeast', 'delicious', 'bookmarks']
    args.dataset_name = dataset_names[0]
    x_train, y_train, _, _ = load_data(args.dataset_name)
    train(x_train, y_train, args)
    #train(x_train, y_train)
