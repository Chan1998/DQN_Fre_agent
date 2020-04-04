import tensorflow as tf
import numpy as np
import collections
import random
import tensorflow.contrib.layers as layers


#ENV = "CartPole-v0"

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
            action_one_hot = tf.one_hot(self.action, self.action_dim)
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
            next_action_one_hot = tf.one_hot(q_next_action, self.action_dim)    #将其转化为one_hot形式
            q_val_next = tf.multiply(q_target, next_action_one_hot)             #将选取动作与targetQ网络做点乘，选出对应动作的价值
            q_val_next = self.sess.run(q_val_next)
            q_target_best_mask = np.max(q_val_next,axis=1)                    #取出该值

        else:
            q_target_best = np.max(q_target, axis=1)
            q_target_best_mask =  q_target_best

        target = reward + self.gamma * q_target_best_mask
        loss, _ = self.sess.run([self.loss, self.train_op],
                                feed_dict={self.inputs_q: state, self.target: target, self.action: action})
        # chose action

    def chose_action(self, current_state):
        current_state = current_state[np.newaxis, :]  # *** array dim: (xx,)  --> (1 , xx) ***
        q = self.sess.run(self.q_value, feed_dict={self.inputs_q: current_state})
        # e-greedy
        if np.random.random() < self.epsilon:
            action_chosen = np.random.randint(0, self.action_dim)
        else:
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


# memory for momery replay
#memory = []
#Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

''''
if __name__ == "__main__":
    #env = gym.make(ENV)
    env = Fre_env

    with tf.Session() as sess:
        DQN = DeepQNetwork(env, sess)
        update_iter = 0
        step_his = []
        for episode in range(EPISODES):
            state = env.reset()
            #env.render()
            reward_all = 0
            success_num = 0
            # training
            for step in range(MAX_STEP):
                action = DQN.chose_action(state)
                #next_state, reward, done, _ = env.step(action)
                if action == 0:
                    success_num +=1
                next_state, reward, done= env.step(state,action)
                reward_all += reward

                if len(memory) > MEMORY_SIZE:
                    memory.pop(0)
                memory.append(Transition(state, action, reward, next_state, float(done)))

                if len(memory) > BATCH_SIZE * 4:
                    batch_transition = random.sample(memory, BATCH_SIZE)
                    # ***
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = map(np.array,
                                                                                                zip(*batch_transition))
                    DQN.train(state=batch_state,
                              reward=batch_reward,
                              action=batch_action,
                              state_next=batch_next_state,
                              done=batch_done
                              )
                    update_iter += 1

                if update_iter % UPDATE_PERIOD == 0:
                    DQN.update_prmt()

                if update_iter % 200 == 0:
                    DQN.decay_epsilon()

                if done:
                    print("[reward_all = {} ] success_num = {}".format(reward_all, success_num))
                    break

                state = next_state
'''''