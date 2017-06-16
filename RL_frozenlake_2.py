#import numpy as np
#import random
#a = np.array([[0,2,3],[4,5,6],[7,9,9]])
##a = np.array([[1],[4],[7]])
##print(np.nonzero(a))
##print(np.amax(a))
#
#print(random.choice(np.nonzero(a == np.amax(a))))

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery' : False}
)
env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n, env.action_space.n])
num_epiosdes = 2000

rList = []

for i in range(num_epiosdes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action = random.choice(np.nonzero(Q[state, :] == np.amax(Q[state, :]))[0])

        new_state, reward, done, _ = env.step(action)

        Q[state, action] = reward + np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rage: " + str(sum(rList)/num_epiosdes))
print(Q)