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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import results_plotter
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

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
from utils import *

project = 'SAC_0129'
name = 'SAC'
cwd = os.getcwd()
print(cwd)

print(os.cpu_count())

# GPU 준비
USE_CUDA = torch.cuda.is_available()
dev = torch.device("cuda:0" if USE_CUDA else "cpu")
print("Using Device:", dev)
torch.cuda.empty_cache()

seed_everything(seed=42)

fmu_filename = 'HEV_TMED_Simulator_WLTC_231005_Check.fmu'
fmu_name = 'HEV_TMED_Simulator_WLTC_231005_Check'
start_time = 0.0
stop_time = 1800.0
step_size = 0.01
soc_init = 67
profile_name = 'wltp_1Hz.csv'

gamma = 0.99
batch_size = 2048
learning_rate = 5e-5
save_dir = f'./model/{project}/'
monitor_dir = f'./monitor/{project}/'
checkpoints_dir = f'./checkpoints/{project}/'
log_dir = f'./logs/{project}/'
board_dir = f'./board/{project}/'
num_cpu = 4
episodes = 10
total_timesteps = batch_size*1
env_id = "HEV"
vec_env = DummyVecEnv([make_env(fmu_filename, project, name, monitor_dir, i) for i in range(num_cpu)])
env = HEV(fmu_filename, test=True, start_time=0.0, step_size=1.0, SoC_coeff=10, BSFC_coeff=0.1, NOx_coeff=1, reward_coeff=1, state_coeff=1, alive_reward=1, project=project, name=name)
eval_env = DummyVecEnv([lambda: Monitor(env, monitor_dir+f"_eval")])
checkpoint_callback = CheckpointCallback(save_freq=1800, save_path=checkpoints_dir, name_prefix="rl_model")
eval_callback = EvalCallback(eval_env, best_model_save_path=save_dir, log_path=log_dir, eval_freq=1800, deterministic=False, render=False)

model = SAC("MlpPolicy", vec_env, verbose=1, device="cuda", batch_size=batch_size, gamma=gamma, learning_rate=learning_rate, tensorboard_log=board_dir)


state = env.reset()
action = np.array([0,0])
done = False
cum_reward = 0.0
a_record = []
r_record = []
s_record = []
t=0
while not done:
    next_state, reward, done, _, info = env.step(action)
    cum_reward += reward
    # a_record.append(np.array(action))
    r_record.append(reward)
    s_record.append(info)
    state = next_state
print(f"total reward: {cum_reward}")

plt.plot(r_record)
plt.show()
# model.learn(total_timesteps = episodes*total_timesteps, callback = [checkpoint_callback, eval_callback])
# del model

# best_model = SAC.load(os.path.join(save_dir, "model", "best_model"))
# mean_reward, std_reward = evaluate_policy(best_model, eval_env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")