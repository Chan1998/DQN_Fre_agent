import numpy as np
import random

Apply_num = 400

Action_space_num = 2
Observation_space_num = 2

#初始化环境
def reset():
    state = np.array([0,0])
    return state

#给出下一步环境
def step(state,action):
    done = 0
    next_state = state
    if action ==0 :
        reward = 10
    else:
        reward = 0
    next_state[0] += 1
    if next_state[0] == Apply_num :
        done = 1
    return next_state,reward,done


