import Fre_env as env
import DQN_agent
import tensorflow as tf
import collections
import numpy as np
import random

#智能体变量
MEMORY_SIZE = 100

EPISODES = 5
#MAX_STEP = 500
BATCH_SIZE = 40
UPDATE_PERIOD = 10  # update target network parameters



#环境变量
Apply_num = 200
M = 10              #可用基站数
K = 20 #可用基站频点 = 动作空间,哪个位置输出最大，选定哪个频点
N = Apply_num             #申请用户数量

#Observation_space_num = M*N*K
#Action_space_num = K



# memory for momery replay
memory = []
Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])


#主函数
if __name__ == "__main__":
    with tf.Session() as sess:
        DQN = DQN_agent.DeepQNetwork(env, sess)
        update_iter = 0
        step_his = []
        for episode in range(EPISODES):
            state,Location_matrix,DQN_Allocation_matrix = env.reset()
            #env.render()
            reward_all = 0
            arg_num = 0
            # training
            for step in range(Apply_num):
                action = DQN.chose_action(state)
                #next_state, reward, done, _ = env.step(action)
                next_state,reward,DQN_Allocation_matrix= env.step(reward_all,arg_num,DQN_Allocation_matrix,
                                                                         action,Location_matrix)
                reward_all += reward
                arg_num += 1
                if len(memory) > MEMORY_SIZE:
                    memory.pop(0)
                memory.append(Transition(state, action, reward, next_state))

                if len(memory) > BATCH_SIZE * 4:
                    batch_transition = random.sample(memory, BATCH_SIZE)
                    # ***
                    batch_state, batch_action, batch_reward, batch_next_state = map(np.array,
                                                                                                zip(*batch_transition))
                    DQN.train(state=batch_state,
                              reward=batch_reward,
                              action=batch_action,
                              state_next=batch_next_state
                              )
                    update_iter += 1

                if update_iter % UPDATE_PERIOD == 0:
                    DQN.update_prmt()
                    #print("[after {}tring,][reward_all = {} ]".format(step, reward_all))

                if update_iter % 200 == 0:
                    DQN.decay_epsilon()

                state = next_state
            print(reward_all)