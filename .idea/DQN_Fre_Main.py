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
MEMORY_SIZE = 60
EPISODES = 1            #不同用户分布情况下重复
MAX_STEP = 200
BATCH_SIZE = 10       #单次训练量大小
UPDATE_PERIOD = 10  # update target network parameters目标网络随训练步数更新周期
Lay_num_list = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16]
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

#T = 20          #测试维度

#T = 20     #隐藏层测试维度

#T = 7

#主函数
if __name__ == "__main__":

    # 用户敏感性实验(少频率)
    '''
    T = 80
    r12 = np.zeros(shape=(T), dtype=float)
    r22 = np.zeros(shape=(T), dtype=float)
    r32 = np.zeros(shape=(T), dtype=float)
    r42 = np.zeros(shape=(T), dtype=float)
    
    K = 5
    M = 20

    for n in range(T):
        N = n + 20
        Apply_num = N
        tf.reset_default_graph()
        memory = []
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])
        with tf.Session() as sess:
            # DQN智能体
            DQN = DQN_agent.DeepQNetwork(env, N, M, K, Lay_num_list, True, True, sess)        #Double,Duling = 1
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
                    # print("[after {}tring,][reward_all = {} ]".format(step, reward_all))

                if update_iter % 60 == 0:
                    DQN.decay_epsilon()

                if arg_num == Apply_num:
                    arg_num = 0

                state = next_state
            r4 = reward_all
            r42[n] = r4
        r1, r2, r3 = Oth.run_process(N, M, K, Location_matrix)
        r12[n] = r1
        r22[n] = r2
        r32[n] = r3
        print(r1, r2, r3, r4)
    k = np.arange(1 + 20 , T + 1 +20)
    plt.plot(k, np.log(r12 + 1e-5), color='r', linestyle=':', marker=None, label='random')
    plt.plot(k, np.log(r22 + 1e-5), color='c', linestyle='-.', marker=None, label='Greedy')
    plt.plot(k, np.log(r32 + 1e-5), color='y', linestyle='-', marker=None, label='Ep_Greedy')
    plt.plot(k, np.log(r42 + 1e-5), color='b', linestyle='-', marker=None, label='DQN')
    plt.legend()
    plt.xlabel("Apply_Num")
    plt.ylabel("H")
    plt.title("Apply_number_influence")
    #plt.savefig("20-100用户,多隐藏层，Double，Duling")
    plt.show()
    '''


    # 基站敏感性实验(少频点)
    '''
    r13 = np.zeros(shape=(T), dtype=float)
    r23 = np.zeros(shape=(T), dtype=float)
    r33 = np.zeros(shape=(T), dtype=float)
    r43 = np.zeros(shape=(T), dtype=float)
    
    K = 5
    N = 100
    Apply_num = N
    for m in range(T):
        M = m + 1 + 10
        Apply_num = N
        tf.reset_default_graph()
        memory = []
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])
        with tf.Session() as sess:
            # DQN智能体
            DQN = DQN_agent.DeepQNetwork(env, N, M, K, Lay_num_list,1, sess)        #Double_DQN = 1
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
                    
                if arg_num == Apply_num:
                    arg_num = 0

                state = next_state
            r4 = reward_all
            r43[m] = r4
        r1, r2, r3 = Oth.run_process(N, M, K, Location_matrix)
        r13[m] = r1
        r23[m] = r2
        r33[m] = r3
        print(r1,r2,r3,r4)

    k = np.arange(1+ 10, T + 1 + 10)
    plt.plot(k, np.log(r13 + 1e-5), color='r', linestyle=':', marker=None, label='random')
    plt.plot(k, np.log(r23 + 1e-5), color='c', linestyle='-.', marker=None, label='Greedy')
    plt.plot(k, np.log(r33 + 1e-5), color='y', linestyle='-', marker=None, label='Ep_Greedy')
    plt.plot(k, np.log(r43 + 1e-5), color='b', linestyle='-', marker=None, label='DQN')
    plt.legend()
    plt.xlabel("Base_Num")
    plt.ylabel("H")
    plt.title("Base_number_influence")
    #plt.savefig("10B")
    plt.show()
    '''

    #少频率选择（动作空间）下频点敏感性分析
    '''
    T = 7
    r11 = np.zeros(shape=(T), dtype=float)
    r21 = np.zeros(shape=(T), dtype=float)
    r31 = np.zeros(shape=(T), dtype=float)
    r41 = np.zeros(shape=(T), dtype=float)
    
    N = 100
    Apply_num = N
    M = 20

    for k in range(T):
        K = k + 2
        tf.reset_default_graph()
        memory = []
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])
        with tf.Session() as sess:
            # DQN智能体
            DQN = DQN_agent.DeepQNetwork(env, N, M, K, Lay_num_list, True, True, sess)        #Double,Duling= 1
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

                if update_iter % 60 == 0:
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

    t = np.arange(1 +2 , T + 1 +2)
    plt.plot(t, np.log(r11 + 1e-5), color='r', linestyle=':', marker=None, label='random')
    plt.plot(t, np.log(r21 + 1e-5), color='c', linestyle='-.', marker=None, label='Greedy')
    plt.plot(t, np.log(r31 + 1e-5), color='y', linestyle='-', marker=None, label='Ep_Greedy')
    plt.plot(t, np.log(r41 + 1e-5), color='b', linestyle='-', marker=None, label='Double_DQN')
    plt.xlabel("Frequence_Num")
    plt.ylabel("H")
    plt.title("Frequence_number_influence")
    plt.legend()
    plt.savefig("频点,多隐藏层,Double，Duling分析")
    plt.show()
    '''

    #网络隐藏层数量设置分析
    '''
    Lay_num_list = [[4096, 2048, 1024, 512, 256, 128, 64, 32, 16],
                    [4096, 1024, 256, 64, 16],
                    [2048, 128, 8],
                    [512]]
    
    r5 = np.zeros(shape=(4,MAX_STEP), dtype=float)

    N = 100
    Apply_num = N
    M = 20
    K = 5
    memory = []
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])
    state_0, Location_matrix, DQN_Allocation_matrix_0 = env.reset(N, M, K)
    #r1, r2, r3 = Oth.run_process(N, M, K, Location_matrix)
    for i in range(4):
        lay_num_list = Lay_num_list[i]
        print(lay_num_list)
        tf.reset_default_graph()
        with tf.Session() as sess:
            # DQN智能体
            DQN = DQN_agent.DeepQNetwork(env, N, M, K, lay_num_list, sess)
            state = state_0
            DQN_Allocation_matrix = DQN_Allocation_matrix_0
            update_iter = 0
            step_his = []
            # for episode in range(EPISODES):            #这里取消循环，后面如果加上别忘了缩进
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
                    print("[after {}tring,][reward_all = {} ]".format(step, reward_all))

                if update_iter % 60 == 0:
                    DQN.decay_epsilon()

                if arg_num == Apply_num:
                    arg_num = 0

                state = next_state
                r4 = reward_all
                r5[i,step] = r4

            #print(r1,r2,r3,r4)
    t = np.arange(1, MAX_STEP + 1)
    plt.plot(t, np.log(r5[0,:] + 1e-5), color='b', linestyle='-', marker=None, label='9_levels')
    plt.plot(t, np.log(r5[1,:] + 1e-5), color='r', linestyle=':', marker=None, label='5_levels')
    plt.plot(t, np.log(r5[2,:] + 1e-5), color='c', linestyle='-.', marker=None, label='3_levels')
    plt.plot(t, np.log(r5[3,:] + 1e-5), color='y', linestyle='-', marker=None, label='1_levels')
    plt.xlabel("Hidden_Layer_Num")
    plt.ylabel("H")
    plt.title("Hidden_Layer_influence")
    plt.legend()
    plt.show()
    # plt.savefig("8F,less_Fre,N100,M20(5)")
    '''

    #double_DQN性能分析
    '''
    r6 = np.zeros(shape=([MAX_STEP,2]), dtype=float)

    N = 100
    Apply_num = N
    M = 20
    K = 5
    state_0, Location_matrix, DQN_Allocation_matrix_0 = env.reset(N, M, K)
    for i in range (2):
        tf.reset_default_graph()
        memory = []
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])       
        with tf.Session() as sess:
            # DQN智能体
            DQN = DQN_agent.DeepQNetwork(env, N, M, K, Lay_num_list, i,True, sess)
            state = state_0
            DQN_Allocation_matrix = DQN_Allocation_matrix_0
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
                    print("[网络{},][after {}tring,][reward_all = {} ]".format(i,step, reward_all))

                if update_iter % 60 == 0:
                    DQN.decay_epsilon()

                if arg_num == Apply_num:
                    arg_num = 0

                state = next_state
                r6[step,i] = reward_all
            r4 = reward_all
            print(r4)


    t = np.arange(1 , MAX_STEP + 1 )
    plt.plot(t, np.log(r6[:,0] + 1e-5), color='r', linestyle=':', marker=None, label='DQN')
    plt.plot(t, np.log(r6[:,1] + 1e-5), color='c', linestyle='-.', marker=None, label='Double_DQN')
    plt.xlabel("steps")
    plt.ylabel("H")
    plt.title("Double_DQN_influence")
    plt.legend()
    plt.savefig("Double_DQN性能分析2")
    plt.show()
    '''

    #_DQN性能分析
    '''
    r7 = np.zeros(shape=([MAX_STEP,2]), dtype=float)

    N = 100
    Apply_num = N
    M = 20
    K = 5
    state_0, Location_matrix, DQN_Allocation_matrix_0 = env.reset(N, M, K)
    for i in range (2):
        tf.reset_default_graph()
        memory = []
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])
        with tf.Session() as sess:
            # DQN智能体
            DQN = DQN_agent.DeepQNetwork(env, N, M, K, Lay_num_list, True, i, sess) #这里i为0或1代表是否使用duling
            state = state_0
            DQN_Allocation_matrix = DQN_Allocation_matrix_0
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
                    print("[网络{},][after {}tring,][reward_all = {} ]".format(i,step, reward_all))

                if update_iter % 60 == 0:
                    DQN.decay_epsilon()

                if arg_num == Apply_num:
                    arg_num = 0

                state = next_state
                r7[step,i] = reward_all
            r4 = reward_all
            print(r4)


    t = np.arange(1 , MAX_STEP + 1 )
    plt.plot(t, np.log(r7[:,0] + 1e-5), color='r', linestyle=':', marker=None, label='DQN')
    plt.plot(t, np.log(r7[:,1] + 1e-5), color='b', linestyle='-.', marker=None, label='Duling_DQN')
    plt.xlabel("steps")
    plt.ylabel("H")
    plt.title("Duling_DQN_influence")
    plt.legend()
    #plt.savefig("Duling_DQN性能分析4")
    plt.show()
    '''

