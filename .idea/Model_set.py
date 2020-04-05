import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns


# #设定超参数
# N = 100             #申请用户数量
# M = 25              #可用基站规模：基站以矩形分布，簇头数量为M，应为完全平方数
# K = 4            #可用基站频点

EPXILONG = 0.3      #设置贪心算法ε值

D = 50             #簇直径，基站覆盖范围D/2（m）
W_AP = 30         #簇头总功率1w，30dBm，W*N
#W = 33             #簇头用户传输功率dBm   （2000mw=2w）=33dBm
#G = 0              #天线增益dB，#计算功率为传输功率乘天线增益，取全向天线，无增益
F = 1000            #载波中心频率（MHz）
B = 10             #载波带宽（MHz）
P =  -24               #dBm簇内通信功率假定为，距离为D/4的接收功率，为30-54,大约-24dBm,0.004mW
n_0 = -174             #正常室温下高斯白噪功率谱密度No=10LogKTB，单位dBm/Hz
n_1 = -94              #频带内噪声单位dBm

#自由空间损耗计算公式：LS：32.45 + 20log（f）MHz + 20log（d）Km  （dB）输入基站距离，输出损耗
def LS_caculate(d, f=F):           #以MHz和m输入
    if (d == 0):                   #计算到自身时，不计算同频干扰
        print("同一用户，无法计算")
    else:
        LS = 32.45 + 20 * (np.log10(f) + np.log10(d/1000))
    return LS                  #以dB 输出

#基站距离计算：假设矩形情况下，输入簇头编号，（目标簇头，干扰簇头，簇头数目，单位距离（m））
def Distance_caculate(m1, m2, m, d=D):
    m = int(np.sqrt(m))
    a = int(m1-1)
    b = int(m2-1)
    x_a = (a // m)+1
    y_a = (a % m)+1
    x_b = (b // m)+1
    y_b = (b % m)+1
    distance = (((x_a - x_b)**2 + (y_a - y_b)**2) ** 0.5) * d
    return distance

#随机生成簇头用户分配表
def Location_matrix_def(n,m,k):
    Location_matrix = np.zeros(shape=(n,m,k),dtype=int)
    for i in range (n):
        Location_matrix[i,int(m * random.random()),0:k] = 1
    return Location_matrix

#计算分配矩阵同频干扰，输入（nmk，分配矩阵，连接矩阵，输出同频干扰计算值，n维,单位为mW）
def I_caculate(n,m,k,Allocation_matrix,Location_matrix):
    I_matrix = np.zeros(shape=(n),dtype=float)
    for i in range (n):     #对所有用户进行计算，用户循环
        done = 0
        for l in range (m):     #找到当前用户对应簇头和其分配的对应频点，  探索基站
            if (Location_matrix[i,l,0] == 1):           #找到对应基站
                if (np.sum(Allocation_matrix[i, l, :]) == 0) :           #用户未分配时直接下一用户
                    break
                for j in range (k):                 #用户分配成功，探索频率
                    if (Allocation_matrix[i,l,j] == 1):         #找到频点，开始计算干扰
                        for x in range (n):
                            for t in range (m):
                                if ((Allocation_matrix[x,t,j]==1) and (x != i) ):           #print("发现同频用户")
                                    I_matrix[i] +=( 10 ** ((W_AP - LS_caculate(Distance_caculate(l+1,t+1,m)))*0.1))
                                    #print(i, l, j, x, t, j, I_matrix[i])
                        done = 1     #print("此用户干扰计算结束")
                        break
                if done :
                    break  #干扰计算结束直接下一用户，print("用户结束")
    return I_matrix

#计算分配矩阵传输数据量            输出单位为Gbps
def R_caculate(n,Allocation_matrix,I_matrix):
    r = 0
    for i in range (n):
        if (np.sum(Allocation_matrix[i,:,:])==1):
            if I_matrix[i] == 0:
                r += B * 100 * np.log2(1 + (10**(P*0.1))/(10**(n_1*0.1)))
            else:
                r += B * 100 * np.log2(1 + (10**(P*0.1))/(I_matrix[i] + 10**(n_1*0.1)))
    return r

#定义随机分配矩阵
def Random_Allocation_def(n,m,k,Location_matrix):
    Random_Allocation_matrix = np.zeros(shape=(n, m, k), dtype=int)
    num = 0
    for i in range(n):
        action = int(k * random.random())
        for l in range(m):
            if Location_matrix[i, l, 0] == 1:
                if (np.sum(Random_Allocation_matrix[:,l,action]) == 0): #相应频段无人使用
                    Random_Allocation_matrix[i, l, action] = 1
                    #print("基站%d范围内%d号用户,频段 %d 成功分配" % (l, i, action))
                    num = num + 1
                    break
                else:
                    #print("对于基站%d范围内%d号用户,频段 %d 已被占用" % (l,i,action))
                    break
        #if np.sum(Random_Allocation_matrix[i,:,:])==0:
            #print("%d号用户,随机分配失败" % ( i))
    return Random_Allocation_matrix,num

#定义传统分配矩阵
def Usual_Allocation_matrix_def(n,m,k,Location_matrix):
    Usual_Allocation_matrix = np.zeros(shape=(n,m,k),dtype=int)
    num = 0
    for i in range(n):
        done = 0
        for l in range(m):
            if Location_matrix[i, l, 0] == 1:   #找到对应基站
                for x in range(k):
                    if (np.sum(Usual_Allocation_matrix[:, l, x]) == 0):  # 相应频段无人使用
                        Usual_Allocation_matrix[i, l, x] = 1
                        # print("基站%d范围内%d号用户,频段 %d 成功分配" % (l, i, x))
                        # print("%d号用户,随机分配成功" % ( i))
                        num = num + 1
                        done = 1
                        break
            if done :
                break
        # if np.sum(Usual_Allocation_matrix[i,:,:])==0:
        #     print("%d号用户,随机分配失败" % ( i))
    return Usual_Allocation_matrix, num

#定义贪心算法分配矩阵
def Greedy_Allocation_matrix_def(n,m,k,Location_matrix):
    Greedy_Allocation_matrix = np.zeros(shape=(n, m, k), dtype=int)
    Greedy_Allocation_matrix_2 = np.zeros(shape=(n, m, k), dtype=int)
    num = 0
    for i in range (n):
        max_R = 0
        next_R = 0
        done = 0
        for l in range(m):
            if Location_matrix[i, l, 0] == 1:       #找到对应基站
                done = 1
                for x in range(k):
                    if np.sum(Greedy_Allocation_matrix[:,l,x]) == 0:
                        Greedy_Allocation_matrix_2[i, l, x] = 1
                        I_matrix = I_caculate(n, m, k, Greedy_Allocation_matrix_2, Location_matrix)
                        next_R = R_caculate(n, Greedy_Allocation_matrix_2, I_matrix)
                        Greedy_Allocation_matrix_2[i, l, x] = 0
                        if (next_R > max_R):
                            Greedy_Allocation_matrix[i, l, :] = 0
                            Greedy_Allocation_matrix[i, l, x] = 1
                            max_R = next_R
                            #print("贪心算法更迭一次")
                Greedy_Allocation_matrix_2[i, l, :] = Greedy_Allocation_matrix[i, l, :]
                    #else:
                        # print("对于基站%d范围内%d号用户,频段 %d 已被占用" % (l, i, x))
                        # print("尝试下一频点")
                        #pass
                #print("该用户分配结束")
            if done:
                break
        if np.sum(Greedy_Allocation_matrix[i, :, :]) == 0:
             #print("%d号用户,贪心分配失败" % (i))
             pass
        else:
            #print("%d号用户,贪心分配成功" % (i))
            num = num +1
    return Greedy_Allocation_matrix,num

#定义ε—贪心算法分配矩阵
def Ep_Greedy_Allocation_matrix_def(n,m,k,Location_matrix):
    Ep_Greedy_Allocation_matrix = np.zeros(shape=(n, m, k), dtype=int)
    Ep_Greedy_Allocation_matrix_2 = np.zeros(shape=(n, m, k), dtype=int)
    num = 0
    epsilong = 0
    for i in range (n):
        max_R = 0
        next_R = 0
        done = 0
        epsilong = random.random()
        for l in range(m):
            if Location_matrix[i, l, 0] == 1:  # 找到对应基站
                done = 1
                if epsilong >= EPXILONG:
                    #print("%d号申请者使用贪心分配" %(i))
                    for x in range(k):
                        if np.sum(Ep_Greedy_Allocation_matrix[:, l, x]) == 0:
                            Ep_Greedy_Allocation_matrix_2[i, l, x] = 1
                            I_matrix = I_caculate(n, m, k, Ep_Greedy_Allocation_matrix_2, Location_matrix)
                            next_R = R_caculate(n, Ep_Greedy_Allocation_matrix_2, I_matrix)
                            Ep_Greedy_Allocation_matrix_2[i, l, x] = 0
                            if (next_R > max_R):
                                Ep_Greedy_Allocation_matrix[i, l, :] = 0
                                Ep_Greedy_Allocation_matrix[i, l, x] = 1
                                max_R = next_R
                                # print("贪心算法更迭一次")
                    Ep_Greedy_Allocation_matrix_2[i, l, :] = Ep_Greedy_Allocation_matrix[i, l, :]
                else:
                    #print("%d号申请者使用随机分配" % (i))
                    for x in range(k):
                        if (np.sum(Ep_Greedy_Allocation_matrix[:, l, x]) == 0):  # 相应频段无人使用
                            Ep_Greedy_Allocation_matrix[i, l, x] = 1
                            # print("基站%d范围内%d号用户,频段 %d 成功分配" % (l, i, x))
                            # print("%d号用户,随机分配成功" % ( i))
                            break
            if done:
                break

        if np.sum(Ep_Greedy_Allocation_matrix[i, :, :]) == 0:
            pass#print("%d号用户,随机分配失败" % (i))
        else:
            #print("%d号用户,随机分配成功" % (i))
            num = num +1
    return Ep_Greedy_Allocation_matrix,num

#用户簇头分布热点图显示函数
def Location_matrix_show(Location_matrix):
    Location_matrix_2D = Location_matrix[:,:,0]
    sns.heatmap(Location_matrix_2D ,annot=False, vmin=0, vmax=1, cmap="Blues", xticklabels=False  ,
                    yticklabels =False   )
    plt.xlabel ("AP_channl(number)")
    plt.ylabel ("Users(person)")
    plt.title("AP_Users_Allocation")
    plt.show()

#不同方法分配矩阵热点图显示函数
def Allocation_matrix_show(n,m,k,Allocation_matrix):
    Allocation_matrix_2D = Allocation_matrix.reshape(n,m * k)
    sns.heatmap(Allocation_matrix_2D ,annot=False, vmin=0, vmax=1, cmap="Greens", xticklabels=False,
                    yticklabels =False   )
    plt.xlabel ("AP_channls")
    plt.ylabel ("Users")
    plt.title("Channls_Allocation")
    plt.show()


def Other_method_process(n,m,k,Location_matrix):
    # print(Location_matrix)
    # 随机算法
    Allocation_matrix1, num1 = Random_Allocation_def(n, m, k, Location_matrix)
    I_matrix1 = I_caculate(n, m, k, Allocation_matrix1, Location_matrix)
    r1 = R_caculate(n, Allocation_matrix1, I_matrix1)

    # 传统算法
    Allocation_matrix2, num2 = Usual_Allocation_matrix_def(n, m, k, Location_matrix)
    I_matrix2 = I_caculate(n, m, k, Allocation_matrix2, Location_matrix)
    r2 = R_caculate(n, Allocation_matrix2, I_matrix2)

    # 贪心算法
    Allocation_matrix3, num3 = Greedy_Allocation_matrix_def(n, m, k, Location_matrix)
    I_matrix3 = I_caculate(n, m, k, Allocation_matrix3, Location_matrix)
    r3 = R_caculate(n, Allocation_matrix3, I_matrix3)

    # ε—贪心算法
    Allocation_matrix4, num4 = Ep_Greedy_Allocation_matrix_def(n, m, k, Location_matrix)
    I_matrix4 = I_caculate(n, m, k, Allocation_matrix4, Location_matrix)
    r4 = R_caculate(n, Allocation_matrix4, I_matrix4)
    return r1, r2, r3, r4, num1, num2, num3, num4

#  if __name__ == "__main__":
    #pass
    # print(Other_method_process(N,M,K,Location_matrix))

    # Allocation_matrix1,num1 = Random_Allocation_def(N, M, K, Location_matrix)
    # I_matrix1 = I_caculate(N, M, K, Allocation_matrix1, Location_matrix)
    # r1 = R_caculate(N,Allocation_matrix1,I_matrix1)
    # print(num1,r1)
    #
    # Allocation_matrix2, num2 = Usual_Allocation_matrix_def(N, M, K, Location_matrix)
    # I_matrix2 = I_caculate(N, M, K, Allocation_matrix2, Location_matrix)
    # r2 = R_caculate(N, Allocation_matrix2, I_matrix2)
    # print(num2, r2)
    #
    # Allocation_matrix3, num3 = Greedy_Allocation_matrix_def(N, M, K, Location_matrix)
    # I_matrix3 = I_caculate(N, M, K, Allocation_matrix3, Location_matrix)
    # r3 = R_caculate(N, Allocation_matrix3, I_matrix3)
    # print(num3, r3)
    #
    # Allocation_matrix4, num4 = Ep_Greedy_Allocation_matrix_def(N, M, K, Location_matrix)
    # I_matrix4 = I_caculate(N, M, K, Allocation_matrix4, Location_matrix)
    # r4 = R_caculate(N, Allocation_matrix4, I_matrix4)
    # print(num4, r4)


    #T = 10


    #r21 = np.zeros(shape=(T), dtype=float)


    # N = 200
    # M = 49
    # K = 5
    # #r11 = np.zeros(shape=(N), dtype=float)
    # Location_matrix = Location_matrix_df(N, M, K)
    # Allocation_matrix1, num1 ,r11= Random_Allocation_def(N, M, K, Location_matrix)
    #
    #
    # k = np.arange(1, N + 1)
    # plt.plot(k, r11, color='b', label='R')
    # #plt.plot(k, r21, color='c', linestyle='-.', marker='o', label='success_num')
    # plt.legend()
    # #plt.figure()
    # plt.xlabel("Step(person)")
    # plt.ylabel("H(Gbps)")
    # plt.title("Success_num_influence")
    # plt.show()
