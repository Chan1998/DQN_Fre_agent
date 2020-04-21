import Model_set as model
import tensorflow as tf
import collections
import numpy as np
import random
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import math
#import seaborn as sns

#  tensorboard --logdir=C:\Users\CP\PycharmProjects\DQN_agent_Fre\.idea\DN\summaries


#智能体变量
MEMORY_SIZE = 5000
EPISODES = 1            #不同用户分布情况下重复
MAX_STEP = 50000
BATCH_SIZE = 32       #单次训练量大小
#UPDATE_PERIOD = 20  # update target network parameters目标网络随训练步数更新周期
#decay_epsilon_STEPS = 100       #降低探索概率次数
Lay_num_list = [640,640,640] #隐藏层节点设置
DATA_SIZE = 5000

N = 20
M = 4
K = 5


class DeepNetwork():
    def __init__(self, n, m, k, lay_num_list,batch_size, sess=None):
        self.batch_size = batch_size
        self.action_dim = n
        self.state_dim =  n
        self.m = m
        self.k = k
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
            #out_action = tf.nn.softmax(out_q)
            return out_q

    # create q_network & target_network
    def network(self):
        self.phase = tf.placeholder(tf.bool, name='phase')
        # network
        self.inputs_state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name="inputs_state")
        scope_var = "network"
        self.output_action = self.net_frame(self.phase, self.lay_num_list, self.inputs_state, self.action_dim, scope_var,
                                      reuse=True)
        self.output_action = tf.clip_by_value(self.output_action, 0, self.k)
        with tf.variable_scope("loss"):
            self.inputs_action = tf.placeholder(dtype=tf.float32, shape=[None,self.action_dim], name="inputs_action")
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits= self.output_action, labels= self.inputs_action)
            tf.summary.histogram('loss', self.loss)
        with tf.variable_scope("train"):
            optimizer = tf.train.RMSPropOptimizer(0.001)
            self.train_op = optimizer.minimize(self.loss)

    def train(self, state, action):
        output= self.sess.run([self.output_action], feed_dict={self.phase:True, self.inputs_state: state })
        output = np.reshape(output, [self.batch_size, self.action_dim])
        summery, _, _ = self.sess.run([self.merged, self.loss, self.train_op],
                                      feed_dict={ self.phase:True,self.output_action: output,
                                                  self.inputs_state: state,
                                                  self.inputs_action: action})
        return summery

    def test(self, state, action):
        output = self.sess.run([self.output_action], feed_dict={self.phase:False, self.inputs_state: state })
        output = np.reshape(output,[1,self.action_dim])
        loss = self.sess.run([self.loss],
                             feed_dict={ self.phase:False, self.output_action: output,
                                         self.inputs_state: state,
                                         self.inputs_action: action})
        Location_dict = {}
        state = np.reshape(state,[self.state_dim])
        action = np.reshape(output, [self.action_dim])
        for i in range(len(action)):
            action[i] = math.floor(action[i])
        print(action)
        for i in range(self.state_dim):
            Location_dict[i] = state[i]
        DQN_dict = {}
        unoccupied_dict = {}
        num = 0
        for l in range(self.m):  # 创建簇头已占用频点集合，记录占用情况
            unoccupied_dict[l] = set(np.arange(0, self.k))
        for i in range(self.state_dim):  # 开始对用户进行分配
            n_l = Location_dict.get(i)  # 查找对应用户的连接簇头
            chose_action = action[i] - 1
            if chose_action in unoccupied_dict[n_l]:
                num += 1
                DQN_dict[i] = chose_action
                unoccupied_dict[n_l].remove(chose_action)
                # print("对于基站%d范围内%d号用户,频段 %d 分配成功" % (n_l,i,action))
                # else:
                #     # print("对应基站没有可用频点剩余，用户分配失败")
                #     pass
        I_dict = model.I_caculate(self.m, Location_dict, DQN_dict)
        reward_all = model.R_caculate(DQN_dict, I_dict)
        del unoccupied_dict
        return DQN_dict, num, reward_all, loss



#主函数
if __name__ == "__main__":

    #设置训练集
    memory = []
    Transition = collections.namedtuple("Transition", ["state", "action"])
    print("开始创建数据")
    for i in range(DATA_SIZE):
        if i % 500 ==1:
            print("创建{}个数据。。。".format(i))
        Location_dict = model.Location_dict_def(N,M)      #创建用户基站对应字典,输入用户数及基站总数量
        Greedy_dict, _ = model.Greedy_dict_def(N, M, K, Location_dict)        #获取分配矩阵
        state = list(Location_dict.values())
        for i in range(N):
            state[i] = float(state[i])
        action = np.zeros(N,float)
        for i in Greedy_dict:
            action[i] = Greedy_dict[i] + 1
        if len(memory) > MEMORY_SIZE:  # 超出长度删除最初经验
            memory.pop(0)
        memory.append(Transition(state, list(action)))  # 储存经验
    print("创建数据完成")

    print("开始训练")
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 固定比例占用显存
    with tf.Session(config=config) as sess:
        DN = DeepNetwork(N, M, K, Lay_num_list, BATCH_SIZE, sess)
        #update_iter = 0
        for step in range(MAX_STEP):
            batch_transition = random.sample(memory, BATCH_SIZE)  # 开始随机抽取
            batch_state, batch_action = map(np.array,zip(*batch_transition))
            summery = DN.train(state=batch_state,  # 进行训练
                                action=batch_action )
            #update_iter += 1
            DN.write.add_summary(summery, step)
            if step % 200 == 0:
                print("进行200次训练。。。")

            if step % 1000 == 0:
                print("进行一次测试：")
                test_transition = random.sample(memory, 1)  # 开始随机抽取
                test_state, test_action = map(np.array, zip(*test_transition))
                _, num, reward_all, loss = DN.test(state=test_state,action=test_action)  # 进行测试
                print("[after {} tring,][success_num is {}][reward_all = {} ]"
                       .format( step, num, reward_all))

        print("进行实验：")
        Location_dict = model.Location_dict_def(N, M)  # 创建用户基站对应字典,输入用户数及基站总数量
        Greedy_dict1, num1 = model.Greedy_dict_def(N, M, K, Location_dict)  # 获取分配矩阵
        I_dict1 = model.I_caculate(M, Location_dict,Greedy_dict1)
        reward_all1 = model.R_caculate(Greedy_dict1, I_dict1)
        state = list(Location_dict.values())
        for i in range(N):
            state[i] = float(state[i])
        action = np.zeros(N, float)
        for i in Greedy_dict1:
            action[i] = Greedy_dict1[i] + 1
        state = np.reshape(state, [1,N])
        action = np.reshape(action, [1, N])
        _, num2, reward_all2, loss = DN.test(state=state, action=action)  # 进行测试
        print("[after {} tring,][success_num is {}][reward_all = {} ]"
              .format(step, num2, reward_all2))
        print("[For Greedy method:][success_num is {}][reward_all = {} ]"
              .format( num1, reward_all1))




