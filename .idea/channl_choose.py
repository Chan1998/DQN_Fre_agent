import numpy as np
import tensorflow as tf
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import tensorflow.contrib.layers as layers
# import os
# os.environ[ "CUDA_VISIBLE_DEVICES"] = "-1"

#build DQN_agent
class DeepQNetwork():
    def __init__(self, state_size, action_size,lay_num_list, sess=None, gamma=0.8, epsilon=0.8, Double_DQN=False, Duling_DQN=True):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_size
        self.state_dim =  state_size
        self.lay_num_list = lay_num_list
        self.double_DQN = Double_DQN
        self.duling_DQN = Duling_DQN
        self.network()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.write = tf.summary.FileWriter("DN/summaries", sess.graph)

    # net_frame using for creating Q & target network
    def net_frame(self, phase, hiddens, inpt, num_actions, scope, reuse=None):
        with tf.variable_scope(scope, reuse=False):
            out = inpt
            #out = layers.batch_norm(inpt, center=True, scale=True, is_training=phase)
            for hidden in hiddens:
                out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
                out = layers.batch_norm(out, center=True, scale=True, is_training=phase)
                out = tf.nn.tanh(out, 'relu')
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
            optimizer = tf.train.AdamOptimizer(0.01)
            self.train_op = optimizer.minimize(self.loss)

            # training

    def train(self, state, reward, action, state_next):
        q_target= self.sess.run(self.q_target,feed_dict={self.phase:1,
                                               self.inputs_target: state_next})

        #拖慢速度，不好用
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
        summery, loss, _ = self.sess.run([self.merged, self.loss, self.train_op],
                                feed_dict={self.phase:1, self.inputs_q: state, self.target: target, self.action: action})
        return summery, loss

    # def train(self, state, reward, action, state_next):
    #     q_next = self.sess.run(self.q_value,feed_dict={self.phase: 1,
    #                                                    self.inputs_q: state_next})
    #
    #     q_target_best = np.max(q_next, axis=1)
    #     q_target_best_mask = q_target_best
    #
    #     target = reward + self.gamma * q_target_best_mask
    #     summery, loss, _ = self.sess.run([self.merged, self.loss, self.train_op],
    #                                      feed_dict={self.phase: 1, self.inputs_q: state, self.target: target,
    #                                                 self.action: action})
    #     return summery, loss
        # chose action

    def chose_action_train(self, current_state):
        current_state = current_state[np.newaxis, :]  # *** array dim: (xx,)  --> (1 , xx) ***
        #print(np.shape(current_state))
        q = self.sess.run(self.q_value, feed_dict={self.phase:1, self.inputs_q: current_state})
        #print(q)
        # e-greedy
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

#创建储存智能体
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


            #读取数据集


#From one state,actions and observations make new states
def state_gen(state,action,obs):
    state_out = state.tolist()
    state_out.append(action)
    state_out.append(obs)
    state_out = state_out[2:]
    return np.asarray(state_out)


# Fetch states,actions,observations and next state from memory
def get_states(batch):
    states = []
    for i in batch:
        states.append(i[0])
    state_arr = np.asarray(states)
    state_arr = state_arr.reshape(32,32)
    #state_arr = state_arr.reshape(8, 32)
    return state_arr

def get_actions(batch):
    actions = []
    for i in batch:
        actions.append(i[1])
    actions_arr = np.asarray(actions)
    actions_arr = actions_arr.reshape(32)
    #actions_arr = actions_arr.reshape(8)
    return actions_arr

def get_rewards(batch):
    rewards = []
    for i in batch:
        rewards.append(i[2])
    rewards_arr = np.asarray(rewards)
    rewards_arr = rewards_arr.reshape(32)
    #rewards_arr = rewards_arr.reshape(8)
    return rewards_arr

def get_next_states(batch):
    next_states = []
    for i in batch:
        next_states.append(i[3])
    next_states_arr = np.asarray(next_states)
    next_states_arr = next_states_arr.reshape(32,32)
    #next_states_arr = next_states_arr.reshape(8, 32)
    return next_states_arr



data_in = pd.read_csv("./dataset/real_data_trace.csv")
data_in = data_in.drop("index",axis=1)
#设定随机种子
np.random.seed(40)

#设定超参数

TIME_SLOTS = 5000
NUM_CHANNELS = 16               # Number of Channels
memory_size = 1000              # Experience Memory Size
batch_size = 32                 # Batch size for loss calculations (M)
eps = 0.1                       # Exploration Probability
action_size = 16                # Action set size
state_size = 32                 # State Size (a_t-1, o_t-1 ,......, a_t-M,o_t-M)
#learning_rate = 1e-2            # Learning rate
#gamma = 0.9                     # Discount Factor
#hidden_size = 50                # Hidden Size (Put 200 for perfectly correlated)
pretrain_length = 16            # Pretrain Set to be known
n_episodes = 10                 # Number of episodes (equivalent to epochs)

decay_epsilon_STEPS = 20       #降低探索概率次数
UPDATE_PERIOD = 20  # update target network parameters目标网络随训练步数更新周期
Lay_num_list = [50,50] #隐藏层节点设置
show_interval = 50  # To see loss trend iteration-wise (Put this 1 to see full trend)

if __name__ == "__main__":
    # 清空计算图
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7    #固定比例占用显存
    #config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    with tf.Session(config=config) as sess:
        DQN = DeepQNetwork( state_size=state_size,action_size=NUM_CHANNELS,lay_num_list=Lay_num_list, sess=sess)
        exp_memory = ExpMemory(in_size=memory_size)
        history_input = deque(maxlen=state_size)            #Input as states

        # Initialise the state of 16 actions and observations with random initialisation

        for i in range(pretrain_length):
            action = np.random.choice(action_size)
            obs = data_in["channel"+str(action)][i]
            history_input.append(action)
            history_input.append(obs)

        #prob_explore = 0.1  # Exploration Probability

        loss_0 = []
        avg_loss = []
        reward_normalised = []

        update_iter = 0
        for episode in range(n_episodes):
            total_rewards = 0
            loss_init = 0

            print("-------------Episode " + str(episode) + "-----------")
            for time in range(len(data_in) - pretrain_length):
            #for time in range(TIME_SLOTS):
                prob_sample = np.random.rand()
                state_in = np.array(history_input)  # Start State
                #print(np.shape(state_in))

                action = DQN.chose_action_train(state_in)  # 通过网络选择对应动作
                #print(action)
                obs = data_in["channel" + str(action)][time + pretrain_length]  # Observe
                next_state = state_gen(state_in, action, obs)  # Go to next state
                reward = obs
                total_rewards += reward  # Total Reward
                exp_memory.add((state_in, action, reward, next_state))  # Add in exp memory
                state_in = next_state
                history_input = next_state

                if (time > state_size or episode != 0):  # If sufficient minibatch is available
                    batch = exp_memory.sample(batch_size)  # Sample without replacement
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
                    loss_init += loss
                    update_iter += 1
                    DQN.write.add_summary(summery, update_iter)
                    if (episode == 0):
                        loss_0.append(loss)

                    if (time % show_interval == 0):
                        print("Loss  at (t=" + str(time) + ") = " + str(loss))

                    if update_iter % UPDATE_PERIOD == 0:
                        DQN.update_prmt()  # 更新目标Q网络
                        #print("更新网络")

                    if time % decay_epsilon_STEPS == 0:
                        DQN.decay_epsilon()  # 随训练进行减小探索力度


                # Plot Display of Loss in episode 0
                if (time == len(data_in) - pretrain_length - 1 and episode == 0):
                #if (time == TIME_SLOTS and episode == 0):
                    plt.plot(loss_0)
                    plt.xlabel("Iteration")
                    plt.ylabel("Q Loss")
                    plt.title('Iteration vs Loss (Episode 0)')
                    plt.show()

            #Average loss
            print("Average Loss: ")
            print(loss_init/(len(data_in)))
            #Average reward observed in full iterations
            print("Total Reward: ")
            print(total_rewards/len(data_in))
            avg_loss.append(loss_init/(len(data_in)))
            reward_normalised.append(total_rewards/len(data_in))

        # See reward and loss trend episode wise

        plt.plot(reward_normalised)
        plt.xlabel("Episode")
        plt.ylabel("Reward Normalised")
        plt.title("Episode vs Reward Normalised")
        plt.show()

        plt.plot(avg_loss)
        plt.xlabel("Episode")
        plt.ylabel("Average Loss")
        plt.title("Episode vs Average Loss")
        plt.show()