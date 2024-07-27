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
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import results_plotter
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from wandb.integration.sb3 import WandbCallback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import MultivariateNormal
from torch.distributions.normal import Normal
from torch import FloatTensor as FT
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader

import tensorflow

from env import *
from policy import *
# from utils import *

project = 'PPO'
name = 'PPO'
cwd = os.getcwd()
print(cwd)
os.system('cls')

fmu_filename = 'HEV_TMED_Simulator_WLTC_231005_Check.fmu'
fmu_name = 'HEV_TMED_Simulator_WLTC_231005_Check'
log = True
save = True
test_learn = True
test_eval = True
start_time = 0.0
stop_time = 1800.0
step_size = 1
soc_init = 67
limitation_coeff = 2
SoC_coeff = 1
BSFC_coeff = 0.1
NOx_coeff = 0.1
reward_coeff = 1
state_coeff = 1
alive_reward = 1
profile_name = 'wltp_1Hz.csv'

save_dir = f'./model/{project}/'
monitor_dir = f'./monitor/{project}/'
checkpoints_dir = f'./checkpoints/{project}/'
log_dir = f'./logs/{project}/'
board_dir = f'./board/{project}/'
num_cpu = 1
env_id = "HEV"

n_actions = 1
learning_rate=5e-5
n_steps=1800
batch_size=180
n_epochs=10
gamma=0.99
gae_lambda=0.97
clip_range=0.1
clip_range_vf=None
normalize_advantage=False
ent_coef=1e-5
vf_coef=0.005
max_grad_norm=0.1
use_sde=False
sde_sample_freq=-1
rollout_buffer_class=None
rollout_buffer_kwargs=None
target_kl=None
stats_window_size=100
tensorboard_log=None
policy_kwargs=None
verbose=1
seed=None
device='auto'
_init_setup_model=True
episodes = 5000
total_timesteps = int(stop_time)*1
config = {"gamma": gamma, "batch_size": batch_size, "learning_rate": learning_rate}

env = HEV(fmu_filename=fmu_filename, log=log, test=test_eval, start_time=start_time, step_size=step_size, 
            limitation_coeff=limitation_coeff, SoC_coeff=SoC_coeff, BSFC_coeff=BSFC_coeff, NOx_coeff=NOx_coeff, reward_coeff=reward_coeff, state_coeff=state_coeff, alive_reward=alive_reward, 
            project=project, name=name, config=config)

state = env.reset()
done = False
action = np.array([0, 0])
s_list = []
while not done:
    state, reward, done, done, info = env.step(action)
    s_list.append(state)
s_list = np.array(s_list)
plt.plot(s_list[:,2]*144* s_list[:,7]/(s_list[:,7]*2+1e-7))

plt.show()