import os
import sys
from glob import glob
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
from fmpy.util import plot_result, download_test_file
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import random
import argparse
from collections import deque
from tqdm import tqdm
import scipy.signal
import time
import random
import copy
import gymnasium as gym
from gymnasium import spaces
import wandb

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import results_plotter
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from wandb.integration.sb3 import WandbCallback

from env import *
os.system('cls')

fmu_filename = 'HEV_TMED_Simulator_WLTC_231005_Check.fmu'
start_time = 0.0
stop_time = 1800.0
step_size = 0.01
soc_init = 67
profile_name = 'wltp_1Hz.csv'
soc_init = 67
limitation_coeff = 2
SoC_coeff = 1
BSFC_coeff = 0.1
NOx_coeff = 0.1
reward_coeff = 1
state_coeff = 1
alive_reward = 1
soc_base = 67/100

wltp = np.squeeze(pd.read_csv('wltp_1Hz.csv'))
wltp = np.int32(np.round(wltp))

vrs = {}
model_description = read_model_description(fmu_filename)
for variable in model_description.modelVariables:
    vrs[variable.name] = variable.valueReference
unzipdir = extract(fmu_filename)
fmu = FMU2Slave(guid=model_description.guid,
               unzipDirectory=unzipdir,
               modelIdentifier=model_description.coSimulation.modelIdentifier,
               instanceName='instance1')

def DP(speed_i, speed_f, soc_init, action, start_time=0.0, step_size=0.01):

    fmu.instantiate()
    fmu.setupExperiment(startTime=start_time)
    fmu.enterInitializationMode()
    fmu.exitInitializationMode()
    
    action_upper_bound = 20000
    action_lower_bound = -20000
    time=0.0
    reward_sum=0.0
    while time<=1:
        instant_veh_speed = np.interp(time, np.array([0,1]), np.array([speed_i, speed_f]))
        a1 = action[0]*action_upper_bound
        a2 = action[0]*action_upper_bound
        a3 = action[1]/2 + 1
        fmu.setReal([vrs['Driver_sVeh_Target_kph'],vrs['SOC_init'],vrs['Engine_on_line'],vrs['Engine_off_line'],vrs['Engine_OOL']], [instant_veh_speed, soc_init, a1, a2, a3])
        fmu.doStep(currentCommunicationPoint=time, communicationStepSize=0.01)

        state = np.array(fmu.getReal(np.arange(39)))
        speed = state[vrs['Driver_sVeh_kph']]/100
        soc = state[vrs['Bat_SOC']]/100
        BSFC = state[vrs['BSFC_g_kWh[1]']]/300
        NOx = state[vrs['NOX_OUT_MDL']]/2
        avg_speed = (speed_i + speed_f) / 2
        if avg_speed < 0.01:
            NOx_wnd = 0
        else:
            NOx_wnd = NOx * 144 / avg_speed
        limitation = -limitation_coeff*(abs(soc_base - soc)>0.1)
        soc_reward = np.log(1 - abs(soc_base - soc))
        bsfc_reward = - BSFC_coeff * BSFC
        nox_reward = - NOx_coeff * NOx
        EURO7 = -3 * (NOx_wnd>0.5)
        reward = alive_reward + soc_reward + bsfc_reward + nox_reward + limitation + EURO7
        reward = reward_coeff * reward

        reward_sum += reward
        time+=0.01

    return round(speed*100), round(soc*100), BSFC*300, NOx*2, reward

#initialize episodic structure
rEpisode=0
rList=[]
num_episodes=2000
episode_max_length=1800

#initialize discount factor, learning rate
gamma=0.99
learning_rate=5e-4

## Epsilon greedy
eps = 1
eps_decay = 0.999

##create action matrix
a1_matrix = np.arange(-1, 2)
a2_matrix = np.arange(-1, 2)
action_matrix = np.zeros([a1_matrix.shape[0]*a2_matrix.shape[0], 2])
for i in range(a1_matrix.shape[0]):
    for j in range(a2_matrix.shape[0]):
        num = i * a2_matrix.shape[0] + j
        action_matrix[num] = np.array([a1_matrix[i], a2_matrix[j]])
        
speed_matrix = np.arange(np.max(wltp)+1) #132
soc_matrix = np.arange(0,101) #101
state_matrix = np.zeros([speed_matrix.shape[0]*soc_matrix.shape[0], 2])
for i in range(speed_matrix.shape[0]):
    for j in range(soc_matrix.shape[0]):
        num = i * soc_matrix.shape[0] + j
        state_matrix[num] = np.array([speed_matrix[i], soc_matrix[j]])

#create Q table
Q=np.zeros([state_matrix.shape[0], action_matrix.shape[0]]) #matrix Q[s,a]

#logging wandb
wandb.login(key='64ece3c392387f62c905a701aab8de7a4110869e')
project = 'QLearning'
name = 'QLearning'
config = {"gamma": gamma, "learning_rate": learning_rate, "eps_decay": eps_decay}
wandb.init(
    project=project,
    name=name,
    config=config
    )

#execute in episodes
for i in tqdm(range(num_episodes)):
    #reset the environment at the beginning of an episode
    soc_init = 67
    rEpisode = 0
    
    for t in range(episode_max_length):
        ###########SELCT ACTION a for state s using Q-values ##################
        # e-greedy action
        if np.random.rand() < eps:
            a = random.randint(0,action_matrix.shape[0]-1)
        else:
            a = np.argmax(Q[s])
        
        speed_i = wltp[t]
        speed_f = wltp[t+1]
        action = action_matrix[a]
        #get new state, reward, done
        try:
            speed, soc, BSFC, NOx, reward = DP(speed_i, speed_f, soc_init, action)
        except:
            print(i, t)
            break
            
        s = speed_matrix[np.where(speed_matrix==speed_i)][0] * soc_matrix.shape[0] + soc_matrix[np.where(soc_matrix==soc_init)][0]
        s1 = speed_matrix[np.where(speed_matrix==speed_f)][0] * soc_matrix.shape[0] + soc_matrix[np.where(soc_matrix==soc)][0]

        ##### update Q(s,a) ############
        # Q-learning update: Bellman Optimal Equation update
        Q[s][a] = Q[s][a] + learning_rate*(reward + gamma*np.max(Q[s1]) - Q[s][a])

        #break if done, reached terminal state 
        if t == episode_max_length-1:
            wandb.log({
              "reward": rEpisode,
            })
            break
            
        soc_init = soc
        rEpisode += reward
    rList.append(rEpisode)
    eps = eps * eps_decay


np.save('Q.npy', Q)
np.save('rList.npy', np.array(rList))

plt.plot(rList)
# plt.plot(Q)
plt.show()
