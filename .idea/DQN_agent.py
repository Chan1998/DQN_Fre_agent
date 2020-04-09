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

##built class for the DQN
class DeepQNetwork():
    def __init__(self, env, n, m, k, lay_num_list, Double_DQN, Duling_DQN, sess=None, gamma=0.8, epsilon=0.8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = k
        self.state_dim = n*m*k
        self.lay_num_list = lay_num_list
        self.double_DQN = Double_DQN
        self.duling_DQN = Duling_DQN
        self.network()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter("DQN/summaries", sess.graph)

    # net_frame using for creating Q & target network
    def net_frame(self, hiddens, inpt, num_actions, scope, reuse=None):
        with tf.variable_scope(scope, reuse=False):
            out = inpt
            for hidden in hiddens:
                out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
            if (self.duling_DQN):       #Duling_DQN将网络结构改变
                out_v = layers.fully_connected(out, num_outputs=1, activation_fn=None)
                out_a = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
                out_q = out_v + (out_a  - tf.reduce_mean(out_a, axis=1, keep_dims=True))
            else:
                out_q = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
            return out_q

    # create q_network & target_network
    def network(self):
        # q_network
        self.inputs_q = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name="inputs_q")
        scope_var = "q_network"
        self.q_value = self.net_frame(self.lay_num_list, self.inputs_q, self.action_dim, scope_var, reuse=True)

        # target_network
        self.inputs_target = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name="inputs_target")
        scope_tar = "target_network"
        self.q_target = self.net_frame(self.lay_num_list, self.inputs_target, self.action_dim, scope_tar)

        with tf.variable_scope("loss"):
            self.action = tf.placeholder(dtype=tf.int32, shape=[None], name="action")
            with tf.device('/cpu:0'):
                action_one_hot = tf.one_hot(self.action, self.action_dim)
            #action_one_hot = tf.one_hot(self.action, self.action_dim)
            q_action = tf.reduce_sum(tf.multiply(self.q_value, action_one_hot), axis=1)

            self.target = tf.placeholder(dtype=tf.float32, shape=[None], name="target")
            self.loss = tf.reduce_mean(tf.square(q_action - self.target))

        with tf.variable_scope("train"):
            optimizer = tf.train.RMSPropOptimizer(0.001)
            self.train_op = optimizer.minimize(self.loss)

            # training

    def train(self, state, reward, action, state_next):
        q, q_target, q_next = self.sess.run([self.q_value, self.q_target,self.q_value],
                                    feed_dict={self.inputs_q: state,
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
        loss, _ = self.sess.run([self.loss, self.train_op],
                                feed_dict={self.inputs_q: state, self.target: target, self.action: action})
        return loss
        # chose action

    def chose_action_train(self, current_state):
        current_state = current_state[np.newaxis, :]  # *** array dim: (xx,)  --> (1 , xx) ***
        q = self.sess.run(self.q_value, feed_dict={self.inputs_q: current_state})
        # e-greedy
        if np.random.random() < self.epsilon:
            action_chosen = np.random.randint(0, self.action_dim)
        else:
            action_chosen = np.argmax(q)

        return action_chosen

    def chose_action_test(self, current_state):
        current_state = current_state[np.newaxis, :]  # *** array dim: (xx,)  --> (1 , xx) ***
        q = self.sess.run(self.q_value, feed_dict={self.inputs_q: current_state})
        print(q)
        action_chosen = np.argmax(q)
        return action_chosen

    # upadate parmerters
    def update_prmt(self):
        q_prmts = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "q_network")
        target_prmts = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target_network")
        self.sess.run([tf.assign(t, q) for t, q in zip(target_prmts, q_prmts)])  # ***
        #print("updating target-network parmeters...")

    def decay_epsilon(self):
        if self.epsilon > 0.03:
            self.epsilon = self.epsilon - 0.02

# DQN初始化环境
def reset(n, m, k):
    Location_matrix = model.Location_matrix_def(n, m, k)
    DQN_Allocation_matrix = np.zeros(shape=(n, m, k), dtype=int)
    state = np.reshape(DQN_Allocation_matrix, [n * m * k])
    return state, Location_matrix, DQN_Allocation_matrix

# DQN_定义DQN分配矩阵（传输速率为目标）
def DQN_Allocation_add(success_num,reward_all,Location_matrix,DQN_Allocation_matrix,action,arg_num,n,m,k):
    n1 = int(arg_num)
    k1 = int(action)
    l = 0
    x = 0
    reward_all_0 = reward_all
    for i in range(m):      #找到用户对应基站
        if Location_matrix[n1, i, 0] == 1 :
            l = i
            #print("找到对应基站",l)
            break

    # 如果用户还未被分配频谱
    if np.sum(DQN_Allocation_matrix[n1, l, :]) == 0 : #且动作频谱刚好未被占用则将频谱对应分配
        if np.sum(DQN_Allocation_matrix[:, l, k1]) == 0:
            DQN_Allocation_matrix[n1, l, k1] = 1
            #print("基站%d范围内%d号用户,频段 %d 成功分配" % (l, n1, k1))
            success_num += 1
            reward_all = model.R_caculate(n,DQN_Allocation_matrix,
                                    model.I_caculate(n,m,k,DQN_Allocation_matrix,Location_matrix))
        else:     #频谱已经被占用，分配失败
            #print("频段已被占用，分配失败")
            pass

    else:   # 如果用户已经被分配，在给出不同分配方案时进行计算，更好的分配方案保留
        #print("用户已经被分配，查找分配频率")
        for i in range(k) :
            if DQN_Allocation_matrix[n1,l,i]:
                x = i
                #print("用户已经被分配，分配频率为x")
                break
        if k1 != x and np.sum(DQN_Allocation_matrix[:,l,k1])== 0:
            #print("分配结果不相同，进行结果比较")
            DQN_Allocation_matrix2 = DQN_Allocation_matrix
            r0 = model.R_caculate(n,DQN_Allocation_matrix,
                                    model.I_caculate(n,m,k,DQN_Allocation_matrix,Location_matrix))
            DQN_Allocation_matrix2[n1,l,x] = 0
            DQN_Allocation_matrix2[n1,l,k1] = 1
            r1 = model.R_caculate(n, DQN_Allocation_matrix2,
                                    model.I_caculate(n, m, k, DQN_Allocation_matrix2, Location_matrix))
            if r1 > r0 :
                #print("更改用户分配方案")
                DQN_Allocation_matrix = DQN_Allocation_matrix2
                reward_all = r1
    r = reward_all - reward_all_0
    return success_num, DQN_Allocation_matrix, reward_all,r

#给出DQN下一步环境
def DQN_step(success_num,reward_all,arg_num,DQN_Allocation_matrix,action,Location_matrix,n,m,k):
    #next_state = state
    success_num,DQN_Allocation_matrix,reward_all,reward = DQN_Allocation_add(success_num, reward_all,Location_matrix,
                                                      DQN_Allocation_matrix, action, arg_num,n,m,k)
    next_state = np.reshape(DQN_Allocation_matrix,[n*m*k])
    return success_num,next_state,reward,DQN_Allocation_matrix




N = 50
M = 16
K = 3


# #智能体变量
MEMORY_SIZE = 1000
EPISODES = 10            #不同用户分布情况下重复
MAX_STEP = 1000
BATCH_SIZE = 1       #单次训练量大小
UPDATE_PERIOD = 50  # update target network parameters目标网络随训练步数更新周期
decay_epsilon_STEPS = 50       #降低探索概率次数
Lay_num_list = [2048, 1024, 256, 128, 64, 32, 16] #隐藏层节点设置

# #DQN训练，测试代码
def DQN_process(Location_matrix1, n, m, k):
    env = model
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9    #固定比例占用显存
    #config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    with tf.Session(config=config) as sess:
        # DQN智能体
        DQN = DeepQNetwork(env, n, m, k, Lay_num_list,True, True, sess)  # Double,Duling= 1
        # 训练
        for episode in range(EPISODES):            #这里取消循环，后面如果加上别忘了缩进
            memory = []
            Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])
            update_iter = 0
            reward_train_list = []
            loss_list = []
            state, Location_matrix, DQN_Allocation_matrix = reset(n, m, k)
            reward_all = 0
            arg_num = 0
            success_num = 0
            print("网络训练中....")
            for step in range(MAX_STEP):
                action = DQN.chose_action_train(state)            #通过网络选择对应动作
                #进行环境交互
                success_num, next_state, reward, DQN_Allocation_matrix = DQN_step \
                             (success_num,reward_all,arg_num,DQN_Allocation_matrix,action,Location_matrix,N,M,K)
                reward_all += reward
                arg_num += 1
                if len(memory) > MEMORY_SIZE:       #超出长度删除最初经验
                    memory.pop(0)
                memory.append(Transition(state, action, reward, next_state))        #储存经验

                if len(memory) > BATCH_SIZE :    #达到训练要求
                    batch_transition = random.sample(memory, BATCH_SIZE)        #开始随机抽取
                    batch_state, batch_action, batch_reward, batch_next_state = map(np.array,
                                                                                    zip(*batch_transition))
                    loss = DQN.train(state=batch_state,            #进行训练
                              reward=batch_reward,
                              action=batch_action,
                              state_next=batch_next_state
                              )
                    update_iter += 1
                    loss_list.append(loss)
                    if step % 100 == 0:
                        print(loss)
                if update_iter % UPDATE_PERIOD == 0:
                    DQN.update_prmt()                   #更新目标Q网络

                if step % (MAX_STEP/10)==1 :
                    print("[in {}espisodes][after {}tring,][chose action is{}][success_num is{}][reward_all = {} ]"
                          .format(episode, step, action, success_num, reward_all))

                if update_iter % decay_epsilon_STEPS == 0:
                    DQN.decay_epsilon()                 #随训练进行减小探索力度

                if arg_num == n:
                    arg_num = 0

                state = next_state
                reward_train_list.append(reward_all)
            print("[after {}tring,][success_num is{}][reward_all = {} ]"
                  .format(MAX_STEP, success_num, reward_all))
        print("训练结束")


        # 测试
        print("网络测试中....")
        # for episode in range(EPISODES):            #这里取消循环，后面如果加上别忘了缩进
        state, _, DQN_Allocation_matrix = reset(n, m, k)
        Location_matrix = Location_matrix1
        reward_test_list = []
        reward_all = 0
        arg_num = 0
        success_num = 0
        for step in range(n):
            action = DQN.chose_action_test(state)  # 通过网络选择对应动作
            # 进行环境交互
            success_num, next_state, reward, DQN_Allocation_matrix = DQN_step \
                (success_num, reward_all, arg_num, DQN_Allocation_matrix, action, Location_matrix, N, M, K)
            reward_all += reward
            arg_num += 1
            print("[for {}User,][given channl is{}][reward_all = {} ]".format(step, action, reward_all))
            state = next_state
            reward_test_list.append(reward_all)
        print(success_num,reward_all)
    return reward_train_list,reward_test_list,loss_list

def Random_process(Location_matrix,n,m,k,Times):
    Random_reward_list = []
    state, _, DQN_Allocation_matrix = reset(n,m,k)
    reward_all = 0
    arg_num = 0
    success_num = 0
    for i in range(n * Times):
        #print(arg_num)
        action = int(K*random.random())
        success_num,next_state,reward,DQN_Allocation_matrix= DQN_step\
            (success_num,reward_all,arg_num,DQN_Allocation_matrix,action,Location_matrix,n,m,k)
        reward_all += reward
        Random_reward_list.append(reward_all)
        arg_num += 1
        if arg_num == n:
            arg_num = 0
    print(success_num,reward_all)
    return Random_reward_list

Location_matrix2 = model.Location_matrix_def(N,M,K)
t1,t2,loss_list= DQN_process(Location_matrix2,N,M,K)
for i in range(len(loss_list)):
    loss_list[i] += 1e-5
t3 = Random_process(Location_matrix2,N,M,K,Times=1)
c1 = np.arange(0,len(t1))
c2 = np.arange(0,len(t2))
c3 = np.arange(0,len(t2))
c4 = np.arange(0,len(loss_list))
plt.subplot(2,2,1)
plt.plot(c1,t1,'b-')
plt.xlabel("(steps)_User_Num")
plt.ylabel("H(Gbps)")
plt.title("Total_rate")
plt.legend(['DQN_Train'])
plt.subplot(2,2,2)
plt.plot(c4,np.log(loss_list),'c-')
plt.xlabel("(steps)")
plt.ylabel("loss")
plt.title("Train_loss")
plt.legend(['Train_loss'])
plt.subplot(2,2,3)
plt.plot(c2,t2,'r-')
plt.plot(c3,t3,'y-')
plt.xlabel("(steps)_User_Num")
plt.ylabel("H(Gbps)")
plt.title("Total_rate")
plt.legend(['DQN','Random'])

plt.show()


