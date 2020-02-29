import Fre_env as env
import DQN_agent
import tensorflow as tf
import collections
import numpy as np
import random

#智能体变量
MEMORY_SIZE = 10000
EPISODES = 500
MAX_STEP = 500
BATCH_SIZE = 32
UPDATE_PERIOD = 200  # update target network parameters

#环境变量
Action_space_num = 2
Observation_space_num = 2



# memory for momery replay
memory = []
Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


#主函数
if __name__ == "__main__":
    with tf.Session() as sess:
        DQN = DQN_agent.DeepQNetwork(env, sess)
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
