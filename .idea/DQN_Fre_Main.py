import Fre_env as env
import DQN_agent
import Other_method as Oth
import tensorflow as tf
import collections
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

#智能体变量
MEMORY_SIZE = 100
EPISODES = 1            #不同用户分布情况下重复
MAX_STEP = 100
BATCH_SIZE = 10         #单次训练量大小
UPDATE_PERIOD = 10  # update target network parameters目标网络随训练步数更新周期
#Lay_num = 100           #隐藏层层数


#环境变量

#N = 100             #申请用户数量
#M = 10              #可用基站数
#K = 20              #可用基站频点 = 动作空间,哪个位置输出最大，选定哪个频点
#Apply_num = N

#Observation_space_num = M*N*K
#Action_space_num = K
# memory for momery replay
#memory = []
#Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])


T = 8      #测试维度

#主函数
if __name__ == "__main__":


    r11 = np.zeros(shape=(T), dtype=float)
    r21 = np.zeros(shape=(T), dtype=float)
    r31 = np.zeros(shape=(T), dtype=float)
    r41 = np.zeros(shape=(T), dtype=float)
    r12 = np.zeros(shape=(T), dtype=float)
    r22 = np.zeros(shape=(T), dtype=float)
    r32 = np.zeros(shape=(T), dtype=float)
    r42 = np.zeros(shape=(T), dtype=float)
    r13 = np.zeros(shape=(T), dtype=float)
    r23 = np.zeros(shape=(T), dtype=float)
    r33 = np.zeros(shape=(T), dtype=float)
    r43 = np.zeros(shape=(T), dtype=float)


    #频率敏感性实验
    '''
    N = 200
    M = 3
    Apply_num = N
    for k in range (T):
        K = k + 1
        tf.reset_default_graph()
        memory = []
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])
        with tf.Session() as sess:
            #DQN智能体
            DQN = DQN_agent.DeepQNetwork(env,N,M,K,sess)
            update_iter = 0
            step_his = []
            #for episode in range(EPISODES):            #这里取消循环，后面如果加上别忘了缩进
            state,Location_matrix,DQN_Allocation_matrix = env.reset(N,M,K)
            #env.render()
            reward_all = 0
            arg_num = 0
            # training
            for step in range(Apply_num):
                action = DQN.chose_action(state)
                #next_state, reward, done, _ = env.step(action)
                next_state,reward,DQN_Allocation_matrix= env.step(reward_all,arg_num,DQN_Allocation_matrix,
                                                                         action,Location_matrix,N,M,K)
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
            r4 = reward_all
            #print(r4)
            r41[k] = r4
        r1,r2,r3 = Oth.run_process(N, M, K, Location_matrix)
        r11[k] = r1
        r21[k] = r2
        r31[k] = r3
    '''

    '''
    # 用户敏感性实验
    K = 100
    M = 3

    for n in range(T):
        N = n + 1
        Apply_num = N
        tf.reset_default_graph()
        memory = []
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])
        with tf.Session() as sess:
            # DQN智能体
            DQN = DQN_agent.DeepQNetwork(env, N, M, K, sess)
            update_iter = 0
            step_his = []
            # for episode in range(EPISODES):            #这里取消循环，后面如果加上别忘了缩进
            state, Location_matrix, DQN_Allocation_matrix = env.reset(N, M, K)
            # env.render()
            reward_all = 0
            arg_num = 0
            # training
            for step in range(Apply_num):
                action = DQN.chose_action(state)
                # next_state, reward, done, _ = env.step(action)
                next_state, reward, DQN_Allocation_matrix = env.step(reward_all, arg_num, DQN_Allocation_matrix,
                                                                     action, Location_matrix, N, M, K)
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
                    # print("[after {}tring,][reward_all = {} ]".format(step, reward_all))

                if update_iter % 200 == 0:
                    DQN.decay_epsilon()

                state = next_state
            r4 = reward_all
            # print(r4)
            r42[n] = r4
        r1, r2, r3 = Oth.run_process(N, M, K, Location_matrix)
        r12[n] = r1
        r22[n] = r2
        r32[n] = r3
    
    '''

    '''
    #基站敏感性实验
    K = 100
    N = 100
    Apply_num = N
    for m in range(T):
        M = m + 1
        Apply_num = N
        tf.reset_default_graph()
        memory = []
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])
        with tf.Session() as sess:
            # DQN智能体
            DQN = DQN_agent.DeepQNetwork(env, N, M, K, sess)
            update_iter = 0
            step_his = []
            # for episode in range(EPISODES):            #这里取消循环，后面如果加上别忘了缩进
            state, Location_matrix, DQN_Allocation_matrix = env.reset(N, M, K)
            # env.render()
            reward_all = 0
            arg_num = 0
            # training
            for step in range(Apply_num):
                action = DQN.chose_action(state)
                # next_state, reward, done, _ = env.step(action)
                next_state, reward, DQN_Allocation_matrix = env.step(reward_all, arg_num, DQN_Allocation_matrix,
                                                                     action, Location_matrix, N, M, K)
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
                    # print("[after {}tring,][reward_all = {} ]".format(step, reward_all))

                if update_iter % 200 == 0:
                    DQN.decay_epsilon()

                state = next_state
            r4 = reward_all
            # print(r4)
            r43[m] = r4
        r1, r2, r3 = Oth.run_process(N, M, K, Location_matrix)
        r13[m] = r1
        r23[m] = r2
        r33[m] = r3



    k = np.arange(1, T + 1)
    
    '''
    '''
    plt.plot(k, np.log(r12 + 1e-5), color='r', linestyle=':', marker=None, label='random')
    plt.plot(k, np.log(r22 + 1e-5), color='c', linestyle='-.', marker=None, label='Greedy')
    plt.plot(k, np.log(r32 + 1e-5), color='y', linestyle='-', marker=None, label='Ep_Greedy')
    plt.plot(k, np.log(r42 + 1e-5), color='b', linestyle='-', marker=None, label='DQN')
    plt.legend()
    plt.xlabel("Apply_Num")
    plt.ylabel("H")
    plt.title("Apply_number_influence")
    plt.savefig("100N")
    '''
    '''
    plt.plot(k, np.log(r13 + 1e-5), color='r', linestyle=':', marker=None, label='random')
    plt.plot(k, np.log(r23 + 1e-5), color='c', linestyle='-.', marker=None, label='Greedy')
    plt.plot(k, np.log(r33 + 1e-5), color='y', linestyle='-', marker=None, label='Ep_Greedy')
    plt.plot(k, np.log(r43 + 1e-5), color='b', linestyle='-', marker=None, label='DQN')
    plt.legend()
    plt.xlabel("Base_Num")
    plt.ylabel("H")
    plt.title("Base_number_influence")
    plt.savefig("10B")
    plt.show()
    '''

    #少频率选择（动作空间）下频点敏感性分析

    N = 100
    Apply_num = N
    M = 20

    for k in range(T):
        K = k + 1

        tf.reset_default_graph()
        memory = []
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])
        with tf.Session() as sess:
            # DQN智能体
            DQN = DQN_agent.DeepQNetwork(env, N, M, K, sess)
            update_iter = 0
            step_his = []
            # for episode in range(EPISODES):            #这里取消循环，后面如果加上别忘了缩进
            state, Location_matrix, DQN_Allocation_matrix = env.reset(N, M, K)
            # env.render()
            reward_all = 0
            arg_num = 0
            # training
            for step in range(MAX_STEP):
                action = DQN.chose_action(state)
                # next_state, reward, done, _ = env.step(action)
                next_state, reward, DQN_Allocation_matrix = env.step(reward_all, arg_num, DQN_Allocation_matrix,
                                                                     action, Location_matrix, N, M, K)
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

                if arg_num == Apply_num:
                    arg_num = 0

                state = next_state
            r4 = reward_all
            # print(r4)
            r41[k] = r4
        r1, r2, r3 = Oth.run_process(N, M, K, Location_matrix)
        r11[k] = r1
        r21[k] = r2
        r31[k] = r3
        print(r1,r2,r3,r4)

    k = np.arange(1, T + 1)
    plt.plot(k, np.log(r11 + 1e-5), color='r', linestyle=':', marker=None, label='random')
    plt.plot(k, np.log(r21 + 1e-5), color='c', linestyle='-.', marker=None, label='Greedy')
    plt.plot(k, np.log(r31 + 1e-5), color='y', linestyle='-', marker=None, label='Ep_Greedy')
    plt.plot(k, np.log(r41 + 1e-5), color='b', linestyle='-', marker=None, label='DQN')
    plt.legend()
    plt.xlabel("Frequence_Num")
    plt.ylabel("H")
    plt.title("Frequence_number_influence")
    #plt.savefig("8F,less_Fre,N100,M20(5)")

    plt.show()

