import numpy as np
import random
import matplotlib.pyplot as plt
#import seaborn as sns


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
noise_0 = -174             #正常室温下高斯白噪功率谱密度No=10LogKTB，单位dBm/Hz
noise_all = -94              #频带内噪声单位dBm

#基站距离和损耗计算：假设矩形情况下，输入簇头编号，（目标簇头，干扰簇头，簇头数目，单位距离（m）,中心频点MH# z）
def LS_caculate(m1, m2, m, d=D, f=F):
    m = int(np.sqrt(m))
    a = int(m1-1)
    b = int(m2-1)
    x_a = (a // m)+1
    y_a = (a % m)+1
    x_b = (b // m)+1
    y_b = (b % m)+1
    distance = (((x_a - x_b)**2 + (y_a - y_b)**2) ** 0.5) * d       #距离输出（m）
    # 自由空间损耗计算公式：LS：32.45 + 20log（f）MHz + 20log（d）Km（dB）
    LS = 32.45 + 20 * (np.log10(f) + np.log10(distance / 1000))
    return LS           #以dB 输出

#随机生成簇头用户分配表
def Location_dict_def(n,m):       #创建用户基站对应字典,输入用户数及基站总数量
    Location_dict = {}
    for i in range (n):             #随机分配用户位置
        Location_dict[i] = int(m * random.random())
    return Location_dict            #返回对应字典，可使用Location_dict.get获取用户位置信息

#计算分配矩阵同频干扰，输入（n, m，位置字典，频率字典，输出同频干扰计算值，n维,单位为mW）
def I_caculate(m, Location_dict, Frequence_dict):
    I_dict = {}
    for i in  Frequence_dict.keys():     #对所有用户进行计算，用户循环
        I_dict[i] = 0
        n_l = Location_dict.get(i)          #找到当前用户对应簇头
        n_k = Frequence_dict.get(i)
        for x in Frequence_dict.keys():                  #开始遍历其他用户寻找同频干扰
            if (x != i) and (Frequence_dict.get(x) == n_k):
                x_l = Location_dict.get(x)
                #print("发现同频用户")
                I_dict[i] += ( 10 ** ((30 - LS_caculate(n_l+1,x_l+1,m))*0.1))               #这里30是W_AP,以mw输出
        #print("此用户干扰计算结束")
    return I_dict

#计算分配矩阵传输数据量            输出单位为Gbps
def R_caculate(Frequence_dict,I_dict):
    r = 0
    for i in Frequence_dict.keys():     #对所有成功分配用户计算传输速率
        if I_dict[i] == 0:
            # 这里-94为noise_all，
            #r += B * 100 * np.log2(1 + (10**(P*0.1))/(10**(-94*0.1)))
            r += 23253.5            #bit/s
        else:
            #r += B * 100 * np.log2(1 + (10**(P*0.1))/(I_dict[i] + 10**(-94*0.1)))
            #10**(-94*0.1) = 4e-10    10**(P*0.1) = 0.003981071705534969
            r += 1000 * np.log2(1 + 4e-3/(I_dict[i] + 4e-10))
    return r        #输出单位为Gbps

#定义随机分配矩阵
def Random_dict_def(n,k,Location_dict):
    Random_dict = {}
    num = 0
    for i in range(n):
        n_l = 0
        occupied = False
        action = int(k * random.random())   #随机选取一个动作
        n_l = Location_dict.get(i)          #查找对应用户的连接簇头
        for x in Random_dict.keys():                 #查找对应矩阵对应频点是否占用
            if x != i and Location_dict.get(x) == n_l and Random_dict.get(x) == action: #相应频段有人使用
                #print("基站%d范围内%d号用户,频段 %d 被占用" % (n_l, i, action))
                occupied = True
                break
        if occupied:
            #print("用户分配失败")
            continue
        else:
            Random_dict[i] = action
            num += 1
            #print("对于基站%d范围内%d号用户,频段 %d 分配成功" % (n_l,i,action))
    return Random_dict,num

#定义传统分配矩阵
def Usual_dict_def(n, m, k, Location_dict):
    Usual_dict = {}
    unoccupied_dict = {}
    num = 0
    for l in range(m):  # 创建簇头已占用频点集合，记录占用情况
        unoccupied_dict[l] = set(np.arange(0,k))
    for i in range(n):          #开始对用户进行分配
        n_l = Location_dict.get(i)      # 查找对应用户的连接簇头
        if len(unoccupied_dict[n_l]):
            action = unoccupied_dict[n_l].pop()  # 剩余可用频点中随机弹出一个值
            num += 1
            Usual_dict[i] = action
            # print("对于基站%d范围内%d号用户,频段 %d 分配成功" % (n_l,i,action))
        # else:
        #     # print("对应基站没有可用频点剩余，用户分配失败")
        #     pass
    del unoccupied_dict
    return Usual_dict, num

#定义贪心算法分配矩阵
def Greedy_dict_def(n,m,k,Location_dict):
    Greedy_dict = {}
    unoccupied_dict = {}
    num = 0
    for l in range(m):  # 创建簇头已占用频点集合，记录占用情况
        unoccupied_dict[l] = set(np.arange(0, k))
    for i in range (n): #开始用户分配
        max_R = 0
        next_R = 0
        action = 0
        n_l = Location_dict.get(i)          #找到用户对应基站
        if len(unoccupied_dict[n_l]) :      #判断基站是否还有可用频点
            for x in unoccupied_dict[n_l]:
                Greedy_dict[i] = x
                next_R = R_caculate(Greedy_dict,I_caculate(m, Location_dict, Greedy_dict))
                if next_R > max_R:
                    action = x
                    max_R = next_R
            unoccupied_dict[n_l].remove(action)
            Greedy_dict[i] = action
            num += 1
            #print("探索结束,弹出最优动作{}".format(Greedy_dict[i]))
        # else:
        #     print("对应基站没有可用频点剩余，用户分配失败")
    del unoccupied_dict
    return Greedy_dict,num

#定义ε—贪心算法分配矩阵
def Ep_Greedy_dict_def(n,m,k,Location_dict):
    Ep_Greedy_dict = {}
    unoccupied_dict = {}
    num = 0
    for l in range(m):  # 创建簇头已占用频点集合，记录占用情况
        unoccupied_dict[l] = set(np.arange(0, k))
    for i in range (n):     #开始用户分配
        n_l = Location_dict.get(i)  # 找到对应基站
        if len(unoccupied_dict[n_l]):  # 判断基站是否还有可用频点
            epsilong = random.random()
            if epsilong <= EPXILONG:
                #print("%d号申请者使用传统分配" %(i))
                action = unoccupied_dict[n_l].pop()  # 剩余可用频点中随机弹出一个值
                num += 1
                Ep_Greedy_dict[i] = action
                #print("对于基站%d范围内%d号用户,频段 %d 分配成功" % (n_l,i,action))
            else:
                # print("%d号申请者使用贪心分配" % (i))
                max_R = 0
                next_R = 0
                for x in unoccupied_dict[n_l]:
                    Ep_Greedy_dict[i] = x
                    next_R = R_caculate(Ep_Greedy_dict, I_caculate(m, Location_dict, Ep_Greedy_dict))
                    if next_R > max_R:
                        action = x
                        max_R = next_R
                unoccupied_dict[n_l].remove(action)
                Ep_Greedy_dict[i] = action
                num += 1
                # print("探索结束,弹出最优动作{}".format(Ep_Greedy_dict[i]))
        # else:
        #     print("%d号用户,分配失败" % (i))
    del unoccupied_dict
    return Ep_Greedy_dict,num

#用户簇头分布热点图显示函数
# def Location_matrix_show(Location_matrix):
#     Location_matrix_2D = Location_matrix[:,:,0]
#     sns.heatmap(Location_matrix_2D ,annot=False, vmin=0, vmax=1, cmap="Blues", xticklabels=False  ,
#                     yticklabels =False   )
#     plt.xlabel ("AP_channl(number)")
#     plt.ylabel ("Users(person)")
#     plt.title("AP_Users_Allocation")
#     plt.show()

#不同方法分配矩阵热点图显示函数
# def Allocation_matrix_show(n,m,k,Allocation_matrix):
#     Allocation_matrix_2D = Allocation_matrix.reshape(n,m * k)
#     sns.heatmap(Allocation_matrix_2D ,annot=False, vmin=0, vmax=1, cmap="Greens", xticklabels=False,
#                     yticklabels =False   )
#     plt.xlabel ("AP_channls")
#     plt.ylabel ("Users")
#     plt.title("Channls_Allocation")
#     plt.show()


def Other_method_process(n,m,k,Location_dict):
    # print(Location_matrix)
    # 随机算法
    Frequence_dict1, num1 = Random_dict_def(N, K, Location_dict)
    # print(Frequence_dict,num)
    I_dict1 = I_caculate(M, Location_dict, Frequence_dict1)
    # print(I_dict1)
    r1 = R_caculate(Frequence_dict1, I_dict1)

    # 传统算法
    Frequence_dict2, num2 = Usual_dict_def(N, M, K, Location_dict)
    # print(Frequence_dict2, num2)
    I_dict2 = I_caculate(M, Location_dict, Frequence_dict2)
    # print(I_dict2)
    r2 = R_caculate(Frequence_dict2, I_dict2)

    # 贪心算法
    Frequence_dict3, num3 = Greedy_dict_def(N, M, K, Location_dict)
    # print(Frequence_dict3, num3)
    I_dict3 = I_caculate(M, Location_dict, Frequence_dict3)
    # print(I_dict3)
    r3 = R_caculate(Frequence_dict3, I_dict3)

    # ε—贪心算法
    Frequence_dict4, num4 = Ep_Greedy_dict_def(N, M, K, Location_dict)
    # print(Frequence_dict4, num4)
    I_dict4 = I_caculate(M, Location_dict, Frequence_dict4)
    # print(I_dict3)
    r4 = R_caculate(Frequence_dict4, I_dict4)
    return r1, r2, r3, r4, num1, num2, num3, num4


# if __name__ == "__main__":
#     N = 150
#     M = 25
#     K = 4
#     Location_dict = Location_dict_def(N, M)
#










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


