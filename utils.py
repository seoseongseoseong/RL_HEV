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

os.system('cls')

project = 'SAC_0423'
name = 'SAC'
cwd = os.getcwd()
print(cwd)

print("cpu available:", os.cpu_count())

# GPU 준비
USE_CUDA = torch.cuda.is_available()
dev = torch.device("cuda:0" if USE_CUDA else "cpu")
print("Using Device:", dev)
torch.cuda.empty_cache()

seed_everything(seed=42)

# import shutil

# shutil.rmtree('wandb')

rList = np.load('rList.npy')
Q = np.load('Q.npy')

plt.plot(rList/1800)
plt.show()

# import itertools

# ent_coef_values = ["1e-5", "1e-4", "auto_1e-5", "auto_1e-4"]
# learning_rate_values = [1e-3, 5e-4, 1e-4]
# batch_size_values = [2048, 4096, 8192]
# tau_values = [0.005]
# gamma_values = [0.99]

# combinations = list(itertools.product(ent_coef_values, learning_rate_values, batch_size_values, tau_values, gamma_values))

# with open("params.txt", "w") as f:
#     for combo in combinations[:36]:  # Ensure only the first 36 combinations are used
#         f.write(" ".join(map(str, combo)) + "\n")


# fmu_filename = 'HEV_TMED_Simulator_WLTC_231005_Check.fmu'
# fmu_name = 'HEV_TMED_Simulator_WLTC_231005_Check'
# log = False
# test_learn = True
# test_eval = True
# start_time = 0.0
# stop_time = 1800.0
# step_size = 1
# soc_init = 67
# limitation_coeff = 2
# SoC_coeff = 10
# BSFC_coeff = 0.1
# NOx_coeff = 0.1
# reward_coeff = 1
# state_coeff = 1
# alive_reward = 1
# profile_name = 'wltp_1Hz.csv'
# buffer_size = 50000
# ent_coef= 'auto_1e-5'
# learning_starts = 100
# tau = 0.005
# gamma = 0.99
# learning_rate = 5e-3
# batch_size = 256
# save_dir = f'./model/{project}/'
# monitor_dir = f'./monitor/{project}/'
# checkpoints_dir = f'./checkpoints/{project}/'
# log_dir = f'./logs/{project}/'
# board_dir = f'./board/{project}/'
# num_cpu = 1 #os.cpu_count()
# episodes = 10000
# total_timesteps = int(stop_time)*1
# env_id = "HEV"
# config = {"gamma": gamma, "batch_size": batch_size, "learning_rate": learning_rate}

# env = HEV(fmu_filename=fmu_filename, log=log, test=test_eval, start_time=start_time, step_size=step_size, 
#             limitation_coeff=limitation_coeff, SoC_coeff=SoC_coeff, BSFC_coeff=BSFC_coeff, NOx_coeff=NOx_coeff, reward_coeff=reward_coeff, state_coeff=state_coeff, alive_reward=alive_reward, 
#             project=project, name=name, config=config)
# action = [1,0]
# done=False
# s = []
# state = env.reset()

# while not done:
#     state, reward, done, _, info = env.step(action)
#     s.append(state)
# s=np.array(s)
# print(len(s))
# plt.plot(s[:,0])
# plt.show()
# print(state)
# print(env.time)
