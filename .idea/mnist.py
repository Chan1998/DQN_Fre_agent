import tensorflow as tf
import collections
import numpy as np
import random
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import math
from tensorflow.examples.tutorials.mnist import input_data
#import seaborn as sns

#  tensorboard --logdir=C:\Users\CP\PycharmProjects\DQN_agent_Fre\.idea\DN\summaries


#智能体变量
EPISODES = 1            #不同用户分布情况下重复
MAX_STEP = 10000
BATCH_SIZE = 32       #单次训练量大小
#UPDATE_PERIOD = 20  # update target network parameters目标网络随训练步数更新周期
#decay_epsilon_STEPS = 100       #降低探索概率次数
Lay_num_list = [640,640,640] #隐藏层节点设置





class DeepNetwork():
    def __init__(self, lay_num_list,batch_size, sess=None):
        self.batch_size = batch_size
        self.action_dim = 10
        self.state_dim =  784
        self.lay_num_list = lay_num_list
        self.network()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.write = tf.summary.FileWriter("DN/summaries", sess.graph)

    # net_frame using for creating  network
    def net_frame(self, phase, hiddens, inpt, num_actions, scope, reuse=None):
        with tf.variable_scope(scope, reuse=False):
            out = inpt
            out = layers.batch_norm(inpt, center=True, scale=True, is_training=phase)
            for hidden in hiddens:
                out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
                out = layers.batch_norm(out, center=True, scale=True, is_training=phase)
                out = tf.nn.relu(out, 'relu')
            out_q = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
            out_q = tf.nn.softmax(out_q)
            return out_q

    # create q_network & target_network
    def network(self):
        self.phase = tf.placeholder(tf.bool, name='phase')
        # network
        self.inputs_state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name="inputs_state")
        scope_var = "network"
        self.output_action = self.net_frame(self.phase, self.lay_num_list, self.inputs_state, self.action_dim, scope_var,
                                      reuse=True)
        #self.output_action = tf.clip_by_value(self.output_action, 0, self.k)
        with tf.variable_scope("loss"):
            self.inputs_action = tf.placeholder(dtype=tf.float32, shape=[None,self.action_dim], name="inputs_y")
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits= self.output_action, labels= self.inputs_action)
            self.correct_prediction = tf.equal(tf.argmax(self.output_action, 1), tf.argmax(self.inputs_action, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            #tf.summary.histogram('loss', self.loss)
            tf.summary.scalar('loss',tf.reduce_mean(self.loss))
        with tf.variable_scope("train"):
            optimizer = tf.train.RMSPropOptimizer(0.0001)
            self.train_op = optimizer.minimize(self.loss)

    def train(self, state, action):
        # print(np.shape(state))
        # print(np.shape(action))
        output= self.sess.run([self.output_action], feed_dict={self.phase:True, self.inputs_state: state })
        output = np.array(output)
        output = np.reshape(output,(-1, self.action_dim))
        # print(np.shape(output))
        summery, _, _ = self.sess.run([self.merged, self.loss, self.train_op],
                                      feed_dict={ self.phase:True,self.output_action: output,
                                                  self.inputs_state: state,
                                                  self.inputs_action: action})
        return summery


    def acc(self,state, action):
        output = self.sess.run([self.output_action], feed_dict={self.phase: False, self.inputs_state: state})
        output = np.array(output)
        output = np.reshape(output, (-1, self.action_dim))
        accuracy_rate = self.sess.run([self.accuracy],
                                      feed_dict={self.phase: False, self.output_action: output,
                                                 self.inputs_state: state,
                                                 self.inputs_action: action})
        return accuracy_rate

#主函数
if __name__ == "__main__":
    #设置训练集
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 固定比例占用显存
    with tf.Session(config=config) as sess:
        print("开始创建数据")
        mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        print("创建数据完成")
        print("开始训练")
        DN = DeepNetwork(Lay_num_list, BATCH_SIZE, sess)
        #update_iter = 0
        acc_list = []

        for step in range(MAX_STEP):        #开始训练
            if step % 100 == 0:        # 进行验证
                print("进行一次验证：")
                validate_acc = DN.acc(state=mnist.validation.images, action=mnist.validation.labels)
                print("After {} training steps, validation accuracy using average model is {}" .format(step, validate_acc))
                acc_list.append(validate_acc)
            # 产生此轮使用的一个batch训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            summery = DN.train(state=xs,
                                action=ys )
            #update_iter += 1
            DN.write.add_summary(summery, step)
            if step % 200 == 0:
                print("进行200次训练。。。")

        print("进行实验：")
        # 训练结束后，在测试数据上检测神经网络模型的最终正确率
        test_acc = DN.acc(state= mnist.test.images,action= mnist.test.labels)
        print("After {} training steps, test accuracy "
              "using average model is {}" .format(MAX_STEP, test_acc))

        c1 = np.arange(0, len(acc_list))
        plt.plot(c1, acc_list, 'c-')
        plt.xlabel("(steps)")
        plt.ylabel("accuracy")
        plt.title("MNIST_accuracy")
        #plt.legend(['Train_loss'])
        plt.show()