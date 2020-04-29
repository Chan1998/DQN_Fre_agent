import tensorflow as tf
import numpy as np
#import collections
from collections import deque
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
    def __init__(self, input_size, output_size, lay_num_list, Duling_DQN, sess=None, gamma=0.8, epsilon=0.8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = output_size
        self.state_dim =  input_size
        self.lay_num_list = lay_num_list
        self.duling_DQN = Duling_DQN
        self.network()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.write = tf.summary.FileWriter("DQN/summaries", sess.graph)

    # net_frame using for creating Q & target network
    def net_frame(self, phase, hiddens, inpt, num_actions, scope, reuse=None):
        with tf.variable_scope(scope, reuse=False):
            out = inpt
            #out = layers.batch_norm(inpt,center=True, scale=True, is_training=phase)
            for hidden in hiddens:
                out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
                out = layers.batch_norm(out, center=True, scale=True, is_training=phase)
                out = tf.nn.relu(out, 'relu')
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
            q_action = tf.reduce_max(self.q_value, axis=1, keepdims=False)
            self.target = tf.placeholder(dtype=tf.float32, shape=[None], name="target")
            self.loss = tf.reduce_mean(tf.square(q_action - self.target))
            tf.summary.scalar('loss', self.loss)
        with tf.variable_scope("train"):
            optimizer = tf.train.AdamOptimizer(0.01)
            self.train_op = optimizer.minimize(self.loss)

            # training

    def train(self, state, reward, action, state_next):
        q_target = self.sess.run(self.q_target,feed_dict={self.phase:1,self.inputs_target: state_next})
        q_target_best = np.max(q_target, axis=1)
        #q_target_best_mask =  q_target_best
        target = reward + self.gamma * q_target_best
        summery, loss, _ = self.sess.run([self.merged, self.loss, self.train_op],
                                feed_dict={self.phase:1, self.inputs_q: state, self.target: target, self.action: action})
        return summery, loss

        # chose action

    def chose_action_train(self, current_state):
        current_state = current_state[np.newaxis, :]  # *** array dim: (xx,)  --> (1 , xx) ***
        q = self.sess.run(self.q_value, feed_dict={self.phase:1, self.inputs_q: current_state})
        if np.random.random() < self.epsilon:
            action_chosen = np.random.randint(0, self.action_dim)
        else:
            action_chosen = np.argmax(q)
        #print(np.argmax(q))
        return action_chosen

    def chose_action_test(self, current_state):
        current_state = current_state[np.newaxis, :]  # *** array dim: (xx,)  --> (1 , xx) ***
        q = self.sess.run(self.q_value, feed_dict={self.phase:0, self.inputs_q: current_state})
        #print(q)
        action_chosen = np.argmax(q)
        #print(np.argmax(q), q)
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

#创建储存体
class ExpMemory():
    def __init__(self, in_size):
        self.buffer_in = deque(maxlen=in_size)

    def add(self, exp):
        self.buffer_in.append(exp)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer_in)), size=batch_size, replace=False)
        res = []
        for i in idx:
            res.append(self.buffer_in[i])
        return res

class env_model():
    def __init__(self, n, m, k, state_size, pretrain_length,location_dict):
        self.n = n
        self.m = m
        self.k = k
        self.state_size = state_size
        self.pretrain_length = pretrain_length
        self.history_input = deque(maxlen=state_size)
        self.Location_dict = location_dict


    # DQN_定义DQN分配矩阵（传输速率为目标）,通过动作和输入状态，给出下一状态
    def model_step(self, arg_num, DQN_dict, unoccupied_dict, action):
        n_l = self.Location_dict.get(arg_num)  # 找到用户对应基站
        if action in unoccupied_dict[n_l]:
            DQN_dict[arg_num] = action
            unoccupied_dict[n_l].remove(action)
            reward = 1
            # print("基站%d范围内%d号用户,频段 %d 成功分配" % (n_l, arg_num, action))
        else:
            # print("选取动作不在可选频点集合")
            reward = 0
        return DQN_dict, unoccupied_dict, reward

    # DQN初始化环境
    def reset(self):
        DQN_dict = {}
        unoccupied_dict = {}
        I_dict = {}
        for l in range(self.m):  # 创建簇头已占用频点集合，记录占用情况
            unoccupied_dict[l] = set(np.arange(0, self.k))
        for i in range(self.pretrain_length):
            action = np.random.choice(self.k)
            DQN_dict, unoccupied_dict, reward = self.model_step(i,DQN_dict,unoccupied_dict,action)
            self.history_input.append(action)
            self.history_input.append(reward)
            state_in = np.array(self.history_input)
        return DQN_dict, unoccupied_dict, state_in

    # From one state,actions and observations make new states
    def state_gen(self, state, action, obs):
        state_out = state.tolist()
        state_out.append(action)
        state_out.append(obs)
        state_out = state_out[2:]
        return np.asarray(state_out)


# reward_all = model.R_caculate(DQN_dict, I_dict)



# Fetch states,actions,observations and next state from memory
def get_states(batch):
    states = []
    for i in batch:
        states.append(i[0])
    state_arr = np.asarray(states)
    state_arr = state_arr.reshape(BATCH_SIZE,state_size)
    #state_arr = state_arr.reshape(8, 32)
    return state_arr

def get_actions(batch):
    actions = []
    for i in batch:
        actions.append(i[1])
    actions_arr = np.asarray(actions)
    actions_arr = actions_arr.reshape(BATCH_SIZE)
    #actions_arr = actions_arr.reshape(8)
    return actions_arr

def get_rewards(batch):
    rewards = []
    for i in batch:
        rewards.append(i[2])
    rewards_arr = np.asarray(rewards)
    rewards_arr = rewards_arr.reshape(BATCH_SIZE)
    #rewards_arr = rewards_arr.reshape(8)
    return rewards_arr

def get_next_states(batch):
    next_states = []
    for i in batch:
        next_states.append(i[3])
    next_states_arr = np.asarray(next_states)
    next_states_arr = next_states_arr.reshape(BATCH_SIZE,state_size)
    #next_states_arr = next_states_arr.reshape(8, 32)
    return next_states_arr




N = 40
M = 4
K = 10


# #智能体变量
MEMORY_SIZE = 80
EPISODES = 50          #不同用户分布情况下重复
MAX_STEP = 40
BATCH_SIZE = 4        #单次训练量大小
UPDATE_PERIOD = 2  # update target network parameters目标网络随训练步数更新周期
decay_epsilon_STEPS = 1       #降低探索概率次数
Lay_num_list = [20,20] #隐藏层节点设置
state_size = 10
pretrain_length = 5



# #DQN训练，测试代码
def DQN_process(Location_dict1, n, m, k,state_size):
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7    #固定比例占用显存
    #config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    with tf.Session(config=config) as sess:
        # DQN智能体
        DQN = DeepQNetwork(state_size,k,Lay_num_list, True, sess)  #  Duling= 1
        exp_memory = ExpMemory(in_size=MEMORY_SIZE)

        # 训练
        # memory = []
        # Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])
        print("网络训练中....")
        update_iter = 0
        loss_0 = []
        for episode in range(EPISODES):
            env = env_model(n, m, k, state_size, pretrain_length, Location_dict1)
            DQN_dict, unoccupied_dict, state_in= env.reset()
            arg_num = 5
            reward_all = 0
            #loss_init = 0
            for step in range(MAX_STEP):
                action = DQN.chose_action_train(state_in)            #通过网络选择对应动作
                #进行环境交互
                DQN_dict, unoccupied_dict, reward = env.model_step(arg_num,DQN_dict,unoccupied_dict,action)
                next_state = env.state_gen(state_in, action, reward)  # Go to next state
                arg_num += 1
                if arg_num == n:
                    arg_num = 0
                reward_all += reward
                #memory.append(Transition(state, action, reward, next_state))        #储存经验
                exp_memory.add((state_in, action, reward, next_state))  # Add in exp memory
                state_in = next_state
                env.history_input = next_state

                if (step > state_size or episode != 0):  # If sufficient minibatch is available
                    batch = exp_memory.sample(BATCH_SIZE)  # Sample without replacement
                    batch_states = get_states(batch)  # Get state,action,reward and next state from memory
                    batch_actions = get_actions(batch)
                    batch_rewards = get_rewards(batch)
                    batch_next_states = get_next_states(batch)
                    # print(np.shape(batch_next_states))
                    # print(np.shape(batch_states))
                    #print(np.shape(batch_rewards))
                    # print(np.shape(batch_actions))
                    summery, loss = DQN.train(state=batch_states,  # 进行训练
                                        reward=batch_rewards,
                                        action=batch_actions,
                                        state_next=batch_next_states
                                        )
                    #loss_init += loss
                    update_iter += 1
                    DQN.write.add_summary(summery, update_iter)
                    loss_0.append(loss)

                    if (step % 20 == 0):
                        print("Loss  at (t=" + str(step) + ") = " + str(loss))

                    if update_iter % UPDATE_PERIOD == 0:
                        DQN.update_prmt()  # 更新目标Q网络
                        #print("更新网络")

                    if step % decay_epsilon_STEPS == 0:
                        DQN.decay_epsilon()  # 随训练进行减小探索力度


                # Plot Display of Loss in episode 0
                if (step == MAX_STEP  and episode == 0):
                #if (time == TIME_SLOTS and episode == 0):
                    plt.plot(loss_0)
                    plt.xlabel("Iteration")
                    plt.ylabel("Q Loss")
                    plt.title('Iteration vs Loss (Episode 0)')
                    plt.show()
            I_dict = model.I_caculate(m, Location_dict1, DQN_dict)
            reward_final = model.R_caculate(DQN_dict, I_dict)
            success_num = len(DQN_dict.keys())
            print("[in {}episode,][success_num is{}][reward_all = {} ]"
                  .format(episode, success_num,reward_final))
        print("训练结束")
    return loss_0

    #     # 测试
    #     print("网络测试中....")
    #     # for episode in range(EPISODES):            #这里取消循环，后面如果加上别忘了缩进
    #     state, DQN_dict, unoccupied_dict, I_dict= reset(n, m, k)
    #     reward_test_list = []
    #     arg_num = 0
    #     reward_all = 0
    #     for step in range(n):
    #         action = DQN.chose_action_test(state)  # 通过网络选择对应动作
    #         # 进行环境交互
    #         next_state, DQN_dict, I_dict, unoccupied_dict, reward, reward_all = DQN_step \
    #             (reward_all, arg_num, Location_dict, DQN_dict, unoccupied_dict, I_dict, action, n, m, k)
    #         arg_num += 1
    #         # I_dict = model.I_caculate(m, Location_dict, DQN_dict)
    #         # reward_all = model.R_caculate(DQN_dict, I_dict)
    #         #print("[for {}User,][given channl is{}][reward_all = {} ]".format(step, action, reward_all))
    #         #print("[for {}User,][given channl is{}]".format(step, action))
    #         state = next_state
    #         reward_test_list.append(reward_all)
    #         success_num = len(DQN_dict.keys())
    #     print(success_num,reward_all,DQN_dict)
    # return reward_test_list

def Random_process(Location_dict,n,m,k, state_size):
    env = env_model(n, m, k, state_size, pretrain_length, Location_dict)
    DQN_dict, unoccupied_dict, state_in = env.reset()
    arg_num = 5
    reward_all = 0
    DQN_dict, unoccupied_dict, state_in = env.reset()
    reward_all = 0
    arg_num = 0
    success_num = 0
    for step in range(MAX_STEP):
        action = np.random.choice(k)  # 通过网络选择对应动作
        # 进行环境交互
        DQN_dict, unoccupied_dict, reward = env.model_step(arg_num, DQN_dict, unoccupied_dict, action)
        next_state = env.state_gen(state_in, action, reward)  # Go to next state
        arg_num += 1
        if arg_num == n:
            arg_num = 0
        reward_all += reward
        # memory.append(Transition(state, action, reward, next_state))        #储存经验
        #exp_memory.add((state_in, action, reward, next_state))  # Add in exp memory
        state_in = next_state
        env.history_input = next_state
    I_dict = model.I_caculate(m, Location_dict1, DQN_dict)
    reward_final = model.R_caculate(DQN_dict, I_dict)
    success_num = len(DQN_dict.keys())
    print(success_num,reward_final)

if __name__ == "__main__":
    Location_dict2 = model.Location_dict_def(N,M)
    t1 = DQN_process(Location_dict2, N, M, K, state_size)
    Random_process(Location_dict2,N,M,K, state_size)
    c1 = np.arange(0,len(t1))
    # c3 = np.arange(0,len(t3))
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
    #plt.plot(c3,t3,'y-')
    # plt.xlabel("(steps)_User_Num")
    # plt.ylabel("H(Gbps)")
    # plt.title("Total_rate")
    # plt.legend(['DQN','Random'])

    plt.show() 
    
"""
if __name__ == "__main__":
    Location_dict2 = model.Location_dict_def(N,M)
    DQN_process(Location_dict2,N,M,K,state_size)
"""