import numpy as np
import random



#定义用户随机分布
def Location_matrix_df(n,m,k):
    Location_matrix = np.zeros(shape=(n,m,k),dtype=int)
    for i in range (n):
        Location_matrix[i,int(m * random.random()),0:k] = 1
    return Location_matrix

#定义DQN分配矩阵（传输速率为目标）
# def DQN_Allocation_add(success_num,reward_all,Location_matrix,DQN_Allocation_matrix,action,arg_num,n,m,k):
#     n1 = int(arg_num)
#     k1 = int(action)
#     #DQN_Allocation_matrix[n1,:,:] = 0
#     flag = 0
#     for l in range(m):
#         if (np.all(Location_matrix[n1, l, 0]) == 1) and (np.all(DQN_Allocation_matrix[n1,l,:] == 0 )):
#             for j in range(n):
#                 flag = flag + DQN_Allocation_matrix[j, l, k1]  # 观察x号频段是否有人使用
#             if flag == 0:
#                 DQN_Allocation_matrix[n1, l, k1] = 1
#                 #print("基站%d范围内%d号用户,频段 %d 成功分配" % (l, n1, k1))
#                 success_num += 1
#                 I_matrix = I_caculate(DQN_Allocation_matrix,n,m,k)
#                 r = R_caculate(DQN_Allocation_matrix,I_matrix,n,m,k)-reward_all
#                 break
#             else:
#                 flag = 0
#                 #print("对于基站%d范围内%d号用户,频段 %d 已被占用" % (l, n1, k1))
#                 #print("基站%d范围内%d号用户,随机分配失败" % (l,n1))
#                 r = 0
#         else:
#             r = 0
#
#     return success_num, DQN_Allocation_matrix,r

#成功人数为目标
def DQN_Allocation_add(success_num,reward_all,Location_matrix,DQN_Allocation_matrix,action,arg_num,n,m,k):
    n1 = int(arg_num)
    k1 = int(action)
    #DQN_Allocation_matrix[n1,:,:] = 0
    flag = 0
    for l in range(m):
        if (np.all(Location_matrix[n1, l, 0]) == 1) and (np.all(DQN_Allocation_matrix[n1,l,:] == 0 )):
            for j in range(n):
                flag = flag + DQN_Allocation_matrix[j, l, k1]  # 观察x号频段是否有人使用
            if flag == 0:
                DQN_Allocation_matrix[n1, l, k1] = 1
                #print("基站%d范围内%d号用户,频段 %d 成功分配" % (l, n1, k1))
                success_num += 1
                I_matrix = I_caculate(DQN_Allocation_matrix,n,m,k)
                r = 0*(R_caculate(DQN_Allocation_matrix,I_matrix,n,m,k)-reward_all) + 1*(3.0)
                break
            else:
                flag = 0
                #print("对于基站%d范围内%d号用户,频段 %d 已被占用" % (l, n1, k1))
                #print("基站%d范围内%d号用户,随机分配失败" % (l,n1))
                r = -0.1
        else:
            r = 0

    return success_num, DQN_Allocation_matrix,r


#计算分配矩阵传输数据量
def R_caculate(Allocation_matrix,I_matrix,n,m,k):
    Allocation_matrix_float = Allocation_matrix.astype(np.float)
    r = np.sum(np.log2(1 + Allocation_matrix_float/(I_matrix/(n/(m*k)) + 0.01)))
    return r

#计算分配矩阵干扰
def I_caculate(Allocation_matrix,n,m,k):
    I_matrix = np.zeros(shape=(n,m,k),dtype=int)
    for l in range (k):
        for i in range (n):
            for j in range (m):
                I_matrix[i,j,l] = np.sum(Allocation_matrix[:,:,l]) - \
                                  np.sum(Allocation_matrix[:,j,l])
    return I_matrix


#初始化环境
def reset(n,m,k):
    Location_matrix = Location_matrix_df(n,m,k)
    DQN_Allocation_matrix = np.zeros(shape=(n, m, k), dtype=int)
    state = np.reshape(DQN_Allocation_matrix,[n*m*k])
    return state,Location_matrix,DQN_Allocation_matrix

#给出下一步环境
def step(success_num,reward_all,arg_num,DQN_Allocation_matrix,action,Location_matrix,n,m,k):
    #next_state = state
    success_num,DQN_Allocation_matrix,reward = DQN_Allocation_add(success_num, reward_all,Location_matrix,
                                                      DQN_Allocation_matrix, action, arg_num,n,m,k)
    next_state = np.reshape(DQN_Allocation_matrix,[n*m*k])
    return success_num,next_state,reward,DQN_Allocation_matrix

'''
state,arg_num ,Location_matrix,DQN_Allocation_matrix= reset()
reward_all = 0
print(np.shape(state))
done = 0
for i in range (Apply_num):
    action = int(K*random.random())
    DQN_Allocation_matrix,reward,done,arg_num = step\
        (arg_num,DQN_Allocation_matrix,action,Location_matrix)
    reward_all += reward
    arg_num += 1
    next_state = np.reshape(DQN_Allocation_matrix, [1, Observation_space_num])
print(np.shape(DQN_Allocation_matrix),np.shape(next_state),reward_all)
'''