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

data_dir = "D:\RDEDB_Stat_TripData\TripDataTintrWithExh"
def make_profile():
    file_names = []
    folder_names = glob(data_dir + "\\*GI")
    for folder_name in folder_names:
        for file_name in glob(folder_name+"/*.csv"):
            file_names.append(file_name)
    files_num = len(file_names)
    done =True
    while done:
        file_num = random.randint(0, files_num)
        file_name = file_names[file_num]
        profile = np.array(pd.read_csv(file_name)['vehSpeed'])
        done = profile.shape[0]==0
    print(f'Profile Name : {file_name},\nProfile Time : {profile.shape[0]}')
    return profile

profile_name = 'wltp_1Hz.csv'
class HEV(gym.Env):
    def __init__(self, fmu_filename, log=True, test=False, start_time=0.0, step_size=0.01, 
                 limitation_coeff=2, SoC_coeff=10, BSFC_coeff=0.1, NOx_coeff=0.1, reward_coeff=1, state_coeff=1, alive_reward=1, project="PPO", name="PPO",
                 config = {"gamma": 0.999, "batch_size": 2048, "learning_rate": 0.0003, "ent_coef": 1e-5}
                ):
        super(HEV, self).__init__()
        self.fmu_filename = fmu_filename
        self.log = log
        self.test = test
        self.start_time = start_time
        self.step_size = step_size
        self.limitation_coeff = limitation_coeff
        self.SoC_coeff = SoC_coeff
        self.BSFC_coeff = BSFC_coeff
        self.NOx_coeff = NOx_coeff
        self.reward_coeff = reward_coeff
        self.state_coeff = state_coeff
        self.alive_reward = alive_reward
        self.time = self.start_time
        self.project = project
        self.name = name
        self.config = config
        if self.test:
            self.vehicle_speed_profile =  np.array(pd.read_csv(profile_name))[:,0]
            self.soc_init = 67/100
#             print(f'Initial SoC : {self.soc_init*100}')
        else:
            self.vehicle_speed_profile = make_profile()
            self.soc_init = random.uniform(57, 77)/100
#             print(f'Initial SoC : {self.soc_init*100}')
        self.time_profile = np.arange(self.vehicle_speed_profile.shape[0])
        self.stop_time = self.vehicle_speed_profile.shape[0] - 1
        self.state_init = np.array([self.soc_init, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.soc_base = 67/100
        self.state = self.state_init
        self.action_upper_bound = 20000
        self.action_lower_bound = -20000
        #self.action_space = [[13500, 2000], [500, 2000], [13500, -15000], [500, -15000]]
        self.actsize = 2
        self.obssize = len(self.state)
        self.vrs = {}
        self.model_description = read_model_description(self.fmu_filename)
        for variable in self.model_description.modelVariables:
            self.vrs[variable.name] = variable.valueReference
        self.vr_input1 = self.vrs['Driver_sVeh_Target_kph']
        self.vr_input2 = self.vrs['SOC_init']
        self.vr_input3 = self.vrs['Engine_on_line']
        self.vr_input4 = self.vrs['Engine_off_line']
        self.vr_input5 = self.vrs['Engine_OOL']
        self.unzipdir = extract(self.fmu_filename)
        self.fmu = FMU2Slave(guid=self.model_description.guid,
                       unzipDirectory=self.unzipdir,
                       modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                       instanceName='instance1')
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.actsize, ), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(self.obssize, ), dtype=np.float32)
        self.metadata = {'render_modes': []}
        self.render_mode = None
        self.reward_range = (-2, 2)
        self.spec = None

        self.episode_reward = 0

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        a1 = action[0]*self.action_upper_bound
        a2 = action[0]*self.action_upper_bound
        a3 = action[1]/2 + 1
        soc_init = self.soc_init*100
        instant_veh_speed = np.interp(self.time, self.time_profile, self.vehicle_speed_profile)
        self.fmu.setReal([self.vr_input1, self.vr_input2, self.vr_input3, self.vr_input4, self.vr_input5], [instant_veh_speed, soc_init, a1, a2, a3]) #input variable, input key(13500 2000)
        self.fmu.doStep(currentCommunicationPoint=self.time, communicationStepSize=self.step_size)
        state = np.array(self.fmu.getReal(np.arange(39))) #5+32
        state_column = np.array([self.vrs['Bat_SOC'], self.vrs['BSFC_g_kWh[1]'], self.vrs['NOX_AVG_WND'], self.vrs['NOX_COR_HOM'], self.vrs['NOX_OUT_MDL'], self.vrs['ObEng_nEng_Rpm'], self.vrs['TrItv_tqEng_Nm'], self.vrs['TrItv_tqP0_Nm'], self.vrs['TrItv_tqP2_Nm'], self.vrs['Driver_sVeh_kph']])
        self.state = state[state_column]
        soc = state[self.vrs['Bat_SOC']]/100
        BSFC = state[self.vrs['BSFC_g_kWh[1]']]/300
        EURO = state[self.vrs['NOX_AVG_WND']]/100
        NOx = state[self.vrs['NOX_OUT_MDL']]/2
        engine_speed = state[self.vrs['ObEng_nEng_Rpm']]/2500
        engine_torque = state[self.vrs['TrItv_tqEng_Nm']]/250
        P0_torque = state[self.vrs['TrItv_tqP0_Nm']]/30
        P2_torque = state[self.vrs['TrItv_tqP2_Nm']]/200
        vehicle_speed = state[self.vrs['Driver_sVeh_kph']]/100
        self.state = np.array([soc, BSFC, EURO, NOx, engine_speed, engine_torque, P0_torque, P2_torque, vehicle_speed], dtype=np.float32)
        
        limitation = -self.limitation_coeff*(abs(self.soc_base - soc)>0.1)
        soc_reward = np.log(1 - abs(self.soc_base - soc))
        bsfc_reward = - self.BSFC_coeff * BSFC
        nox_reward = - self.NOx_coeff * NOx
        reward = self.alive_reward + soc_reward + bsfc_reward + nox_reward + limitation
        reward = self.reward_coeff * reward
        is_done = lambda time: time >= self.stop_time
        info = state[np.array([self.vrs['Bat_SOC'], self.vrs['BSFC_g_kWh[1]'], self.vrs['NOX_AVG_WND'], self.vrs['NOX_COR_HOM'], self.vrs['NOX_OUT_MDL'], self.vrs['ObEng_nEng_Rpm'], self.vrs['TrItv_tqEng_Nm'], self.vrs['TrItv_tqP0_Nm'], self.vrs['TrItv_tqP2_Nm']])]
        info = {
            "info": info
        }
        self.time += self.step_size
        done = is_done(self.time)
        self.episode_reward += reward
        if self.log:
            wandb.log({
                "reward": reward,
                "SoC": soc_reward+limitation,
                "BSFC": bsfc_reward,
                "NOx": nox_reward,
                "action1": action[0],
                "action2": action[1],
            })
            if done:
                wandb.log({
                    "mean reward": self.episode_reward / self.stop_time,
                })
                self.episode_reward = 0
        return self.state*self.state_coeff, reward, done, done, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=self.start_time)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
        self.state = self.state_init
        self.time = self.start_time
        if self.test:
            self.vehicle_speed_profile =  np.array(pd.read_csv(profile_name))[:,0]
            self.soc_init = 67/100
            # print(f'Initial SoC : {self.soc_init*100}')
        else:
            self.vehicle_speed_profile = make_profile()
            self.soc_init = random.uniform(57, 77)/100
            # print(f'Initial SoC : {self.soc_init*100}')
        self.time_profile = np.arange(self.vehicle_speed_profile.shape[0])
        self.stop_time = self.vehicle_speed_profile.shape[0] - 1

        # self.step(np.array([1,0]))

        info = {
            "info": 'env reset'
        }
        return self.state*self.state_coeff, info
    
    def render(self):
        pass
        
    def close(self):
        pass
    
    @property
    def unwrapped(self):
        return self

    @property
    def np_random(self):
        return np.random.default_rng()
    

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def make_env(fmu_filename, log=True, test=False, start_time=0.0, step_size=1.0, 
             limitation_coeff=2, SoC_coeff=10, BSFC_coeff=0.1, NOx_coeff=0.1, reward_coeff=0.5, state_coeff=1, alive_reward=1, 
             project='SAC', name='SAC', config = {"gamma": 0.999, "batch_size": 2048, "learning_rate": 0.0003, "ent_coef": 1e-5}, monitor_dir=f'./monitor/{SAC}/', seed: int = 0):
    def _init():
        env = HEV(fmu_filename=fmu_filename, log=log, test=test, start_time=start_time, step_size=step_size, 
                  limitation_coeff=limitation_coeff ,SoC_coeff=SoC_coeff, BSFC_coeff=BSFC_coeff, NOx_coeff=NOx_coeff, reward_coeff=reward_coeff, state_coeff=state_coeff, alive_reward=alive_reward, 
                  project=project, name=name, config=config)
        env.reset()
        return env
    set_random_seed(seed)
    return _init