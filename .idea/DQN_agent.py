import tensorflow as tf
import numpy as np
import collections
import random
import Model_set as model
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
# import os
# os.environ[ "CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

#MEMORY_SIZE = 10000
#EPISODES = 500
#MAX_STEP = 500
#BATCH_SIZE = 32
#UPDATE_PERIOD = 200  # update target network parameters
#lay_num = 200

# tensorboard --logdir=C:\Users\CP\PycharmProjects\DQN_agent_Fre\.idea\DQN\summaries

##built class for the DQN
class DeepQNetwork():
    def __init__(self, n, m, k, lay_num_list, Double_DQN, Duling_DQN, sess=None, gamma=0.2, epsilon=0.8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = k
        self.state_dim =  n + k
        self.lay_num_list = lay_num_list
        self.double_DQN = Double_DQN
        self.duling_DQN = Duling_DQN
        self.network()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.write = tf.summary.FileWriter("DQN/summaries", sess.graph)

    # net_frame using for creating Q & target network
    def net_frame(self, phase, hiddens, inpt, num_actions, scope, reuse=None):
        with tf.variable_scope(scope, reuse=False):
            #out = inpt
            out = layers.batch_norm(inpt,center=True, scale=True, is_training=phase)
            for hidden in hiddens:
                out1 = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
                out2 = layers.batch_norm(out1, center=True, scale=True, is_training=phase)
                out = tf.nn.tanh(out2, 'relu')
            if (self.duling_DQN):       #Duling_DQN将网络结构改变
                out_v = layers.fully_connected(out, num_outputs=1, activation_fn=None)
                out_a = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
                out_q = out_v + (out_a  - tf.reduce_mean(out_a, axis=1, keepdims=True))
            else:
                out_q = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
            return out_q

    # create q_network & target_network
    def network(self):
        self.phase = tf.placeholder(tf.bool, name='phase')
        # q_network
        self.inputs_q = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name="inputs_q")
        scope_var = "q_network"
        self.q_value = self.net_frame(self.phase, self.lay_num_list, self.inputs_q, self.action_dim, scope_var, reuse=True)

        # target_network
        self.inputs_target = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name="inputs_target")
        scope_tar = "target_network"
        self.q_target = self.net_frame(self.phase, self.lay_num_list, self.inputs_target, self.action_dim, scope_tar)

        with tf.variable_scope("loss"):
            self.action = tf.placeholder(dtype=tf.int32, shape=[None], name="action")
            with tf.device('/cpu:0'):
                action_one_hot = tf.one_hot(self.action, self.action_dim)
            #action_one_hot = tf.one_hot(self.action, self.action_dim)
            q_action = tf.reduce_sum(tf.multiply(self.q_value, action_one_hot), axis=1)

            self.target = tf.placeholder(dtype=tf.float32, shape=[None], name="target")
            self.loss = tf.reduce_mean(tf.square(q_action - self.target))
            tf.summary.scalar('loss', self.loss)
        with tf.variable_scope("train"):
            optimizer = tf.train.RMSPropOptimizer(0.01)
            self.train_op = optimizer.minimize(self.loss)

            # training

    def train(self, state, reward, action, state_next):
        q, q_target, q_next = self.sess.run([self.q_value, self.q_target,self.q_value],
                                    feed_dict={self.phase:1,
                                               self.inputs_q: state,
                                               self.inputs_target: state_next,
                                               self.inputs_q: state_next})


        if (self.double_DQN) :        #doubleDQN需要得到Q网络中的动作所在位置最大索引
            #q_next_axis = np.argmax(q_next,axis=1)
            #q_target_best = q_target[q_next_axis]
            q_next_action = np.argmax(q_next,axis=1)            #找到q网络中最大动作价值索引
            with tf.device('/cpu:0'):
                next_action_one_hot = tf.one_hot(q_next_action, self.action_dim)    #将其转化为one_hot形式
            #next_action_one_hot = tf.one_hot(q_next_action, self.action_dim)
            q_val_next = tf.multiply(q_target, next_action_one_hot)             #将选取动作与targetQ网络做点乘，选出对应动作的价值
            q_val_next = self.sess.run(q_val_next)
            q_target_best_mask = np.max(q_val_next,axis=1)                    #取出该值

        else:
            q_target_best = np.max(q_target, axis=1)
            q_target_best_mask =  q_target_best

        target = reward + self.gamma * q_target_best_mask
        summery, _, _ = self.sess.run([self.merged, self.loss, self.train_op],
                                feed_dict={self.phase:1, self.inputs_q: state, self.target: target, self.action: action})
        return summery

        # chose action

    def chose_action_train(self, current_state):
        current_state = current_state[np.newaxis, :]  # *** array dim: (xx,)  --> (1 , xx) ***
        q = self.sess.run(self.q_value, feed_dict={self.phase:1, self.inputs_q: current_state})
        #print(q)
        # e-greedy
        if np.random.random() < self.epsilon:
            action_chosen = np.random.randint(0, self.action_dim)
        else:
            action_chosen = np.argmax(q)
        print(np.argmax(q))
        return action_chosen

    def chose_action_test(self, current_state):
        current_state = current_state[np.newaxis, :]  # *** array dim: (xx,)  --> (1 , xx) ***
        q = self.sess.run(self.q_value, feed_dict={self.phase:0, self.inputs_q: current_state})
        #print(q)
        action_chosen = np.argmax(q)
        print(np.argmax(q), q)
        return action_chosen

    # upadate parmerters
    def update_prmt(self):
        q_prmts = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "q_network")
        target_prmts = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target_network")
        self.sess.run([tf.assign(t, q) for t, q in zip(target_prmts, q_prmts)])  # ***
        print("updating target-network parmeters...")

    def decay_epsilon(self):
        if self.epsilon > 0.03:
            self.epsilon = self.epsilon - 0.02

# DQN初始化环境
def reset(n, m, k):
    DQN_dict = {}
    unoccupied_dict = {}
    I_dict = {}
    for l in range(m):  # 创建簇头已占用频点集合，记录占用情况
        unoccupied_dict[l] = set(np.arange(0,k))
    #state = DQN_input_def(n, k, set(np.arange(0,k)), I_dict)
    state = DQN_input_def(n, k, set(np.arange(0, k)), DQN_dict)
    return state, DQN_dict, unoccupied_dict, I_dict

#定义DQN网络输入形式 用户可使用频点状态（K个bool形式）+ （所用用户对应干扰情况N个，float）
def DQN_input_def(n, k, unoccupied_frequence_set, DQN_dict):
    frequence_list = np.zeros(shape=(k), dtype=float)
    DQN_list = np.zeros(shape=(n), dtype=float)
    for i in unoccupied_frequence_set:      #先将set转换为列表
        frequence_list[i] = 1
    for i in DQN_dict.keys():
        #DQN_list[i] = 10000 * I_dict.get(i)
        DQN_list[i] = 1
    input_list = np.hstack((frequence_list,DQN_list))
    return input_list


# DQN_定义DQN分配矩阵（传输速率为目标）,通过动作和输入状态，给出下一状态
def DQN_step(reward_all, arg_num, Location_dict, DQN_dict, unoccupied_dict, I_dict, action, n, m, k):
    reward_all0 = reward_all
    n_l = Location_dict.get(arg_num)   #找到用户对应基站
    if action  in unoccupied_dict[n_l]:
        DQN_dict[arg_num] = action
        unoccupied_dict[n_l].remove(action)
        #print("选择对应频点，进行干扰计算更新")
        I_dict = model.I_caculate(m, Location_dict, DQN_dict)
        reward_all = model.R_caculate(DQN_dict,I_dict)
        reward = 0.001 * (reward_all - reward_all0)
        #print("基站%d范围内%d号用户,频段 %d 成功分配" % (n_l, arg_num, action))
    else:
        #print("选取动作不在可选频点集合")
        if arg_num > 0.8 * m*k :
            reward = 0
        else: reward = -8
    next_state = DQN_input_def(n, k, unoccupied_dict[n_l], DQN_dict)
    return next_state, DQN_dict, I_dict, unoccupied_dict, reward, reward_all

#给出DQN下一步环境

# reward_all = model.R_caculate(DQN_dict, I_dict)

'''

N = 20
M = 4
K = 5


# #智能体变量
MEMORY_SIZE = 800
EPISODES = 50          #不同用户分布情况下重复
MAX_STEP = 20
BATCH_SIZE = 1        #单次训练量大小
UPDATE_PERIOD = 5  # update target network parameters目标网络随训练步数更新周期
decay_epsilon_STEPS = 100       #降低探索概率次数
Lay_num_list = [25,16,8,4] #隐藏层节点设置

# #DQN训练，测试代码
def DQN_process(Location_dict1, n, m, k):
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7    #固定比例占用显存
    #config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    with tf.Session(config=config) as sess:
        # DQN智能体
        DQN = DeepQNetwork(n, m, k, Lay_num_list,True, True, sess)  # Double,Duling= 1
        # 训练
        memory = []
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])
        Location_dict = Location_dict1
        print("网络训练中....")
        update_iter = 0
        for episode in range(EPISODES):

            state, DQN_dict, unoccupied_dict, I_dict = reset(n, m, k)
            arg_num = 0
            reward_all = 0
            for step in range(MAX_STEP):
                action = DQN.chose_action_train(state)            #通过网络选择对应动作
                #进行环境交互
                next_state, DQN_dict, I_dict, unoccupied_dict, reward, reward_all = DQN_step \
                        (reward_all ,arg_num, Location_dict, DQN_dict, unoccupied_dict, I_dict, action, n, m, k)
                arg_num += 1
                if len(memory) > MEMORY_SIZE:       #超出长度删除最初经验
                    memory.pop(0)
                memory.append(Transition(state, action, reward, next_state))        #储存经验

                if len(memory) > BATCH_SIZE :    #达到训练要求
                    batch_transition = random.sample(memory, BATCH_SIZE)        #开始随机抽取
                    batch_state, batch_action, batch_reward, batch_next_state = map(np.array,
                                                                                    zip(*batch_transition))
                    summery = DQN.train(state=batch_state,            #进行训练
                              reward=batch_reward,
                              action=batch_action,
                              state_next=batch_next_state
                              )
                    update_iter += 1
                    DQN.write.add_summary(summery,update_iter )
                    # if step % 10 == 0:
                    #     print(loss)
                if update_iter % UPDATE_PERIOD == 0:
                    DQN.update_prmt()                   #更新目标Q网络
                    #print("更新网络",step)

                # if step % 20==0 :
                #     print("[in {}espisodes][after {}tring,][chose action is{}][reward_all = {} ]"
                #           .format(episode, step, action, success_num, reward_all))

                if update_iter % decay_epsilon_STEPS == 0:
                    DQN.decay_epsilon()                 #随训练进行减小探索力度

                state = next_state

                if arg_num == N:
                    arg_num = 0
            reward_all = model.R_caculate(DQN_dict, I_dict)
            success_num = len(DQN_dict.keys())
            print("[in {}episode,][success_num is{}][reward_all = {} ]"
                  .format(episode, success_num, reward_all))
        print("训练结束")


        # 测试
        print("网络测试中....")
        # for episode in range(EPISODES):            #这里取消循环，后面如果加上别忘了缩进
        state, DQN_dict, unoccupied_dict, I_dict= reset(n, m, k)
        reward_test_list = []
        arg_num = 0
        reward_all = 0
        for step in range(n):
            action = DQN.chose_action_test(state)  # 通过网络选择对应动作
            # 进行环境交互
            next_state, DQN_dict, I_dict, unoccupied_dict, reward, reward_all = DQN_step \
                (reward_all, arg_num, Location_dict, DQN_dict, unoccupied_dict, I_dict, action, n, m, k)
            arg_num += 1
            # I_dict = model.I_caculate(m, Location_dict, DQN_dict)
            # reward_all = model.R_caculate(DQN_dict, I_dict)
            #print("[for {}User,][given channl is{}][reward_all = {} ]".format(step, action, reward_all))
            #print("[for {}User,][given channl is{}]".format(step, action))
            state = next_state
            reward_test_list.append(reward_all)
            success_num = len(DQN_dict.keys())
        print(success_num,reward_all,DQN_dict)
    return reward_test_list

def Random_process(Location_dict,n,m,k):
    state,DQN_dict, unoccupied_dict, I_dict = reset(n, m, k)
    Random_reward_list = []
    reward_all = 0
    arg_num = 0
    success_num = 0
    for i in range(n):
        #print(arg_num)
        action = int(K*random.random())
        next_state, DQN_dict, I_dict, unoccupied_dict, reward, reward_all = DQN_step \
            (reward_all, arg_num, Location_dict, DQN_dict, unoccupied_dict, I_dict, action, n, m, k)
        arg_num += 1
        # I_dict = model.I_caculate(m, Location_dict, DQN_dict)
        # reward_all = model.R_caculate(DQN_dict, I_dict)
        Random_reward_list.append(reward_all)
        #print(reward)
        state = next_state
    success_num = len(DQN_dict.keys())
    print(success_num,reward_all)
    return Random_reward_list

if __name__ == "__main__":
    Location_dict2 = model.Location_dict_def(N,M)
    t1= DQN_process(Location_dict2,N,M,K)
    t3 = Random_process(Location_dict2,N,M,K)
    c1 = np.arange(0,len(t1))
    c3 = np.arange(0,len(t3))
    # plt.subplot(2,2,1)
    # plt.plot(c1,t1,'b-')
    # plt.xlabel("(steps)_User_Num")
    # plt.ylabel("H(Gbps)")
    # plt.title("Total_rate")
    # plt.legend(['DQN_Train'])
    # plt.subplot(2,1,1)
    # plt.plot(c4,np.log(loss_list),'c-')
    # plt.xlabel("(steps)")
    # plt.ylabel("loss")
    # plt.title("Train_loss")
    # plt.legend(['Train_loss'])
    # plt.subplot(2,1,2)
    plt.plot(c1,t1,'r-')
    plt.plot(c3,t3,'y-')
    plt.xlabel("(steps)_User_Num")
    plt.ylabel("H(Gbps)")
    plt.title("Total_rate")
    plt.legend(['DQN','Random'])

    plt.show() 
    
'''
