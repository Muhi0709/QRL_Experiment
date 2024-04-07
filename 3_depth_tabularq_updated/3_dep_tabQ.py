import sys
from copy import copy,deepcopy

import logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler("./logs5/log.log", "a")
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


import wandb

wandb.login(key="63cea33f1c9c226a7fdeffedfc683b2b4398cfb5")


import matplotlib.pyplot as plt
from matplotlib import colors

import os
import math
from time import time
import random
from collections import deque, namedtuple
from itertools import count
import json

# from utils import plot_line_graph,visualisation, create_one_hot,custom_two_local

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import f1_score

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit import QuantumCircuit
from qiskit_machine_learning.algorithms import VQC
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.circuit import (
    Parameter,
    QuantumRegister,
    Qubit,
    CircuitInstruction,
    Instruction,
)
from qiskit.circuit.library import TwoLocal

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym as gym
from gym.spaces import Box, Discrete
from gym.envs.registration import register

from PIL import Image, ImageSequence
from PIL.GifImagePlugin import GifImageFile
from typing import List
import io
import os




code_start = time()

d1 = {#rx,ry,cx
    0: 0, 1: 2, 2: 4, 3: 1, 4: 3, 5: 5,
    6: (0,1), 7: (1,2), 8: (3,4), 9: (4,5) 
}
        
d2 = {#ry,rz,cx
    0: 0, 1: 2, 2: 4, 3: 1, 4: 3, 5: 5,
    6: (1,0), 7: (2,1), 8: (4,3), 9: (5,4)
}
        
d3 = {#rz,rx,cx
    0: 0, 1: 2, 2: 4, 3: 1, 4: 3, 5: 5,
    6: (0,2), 7: (2,0), 8: (3,5), 9: (5,3)
}


def plot_line_graph(function_outputs, xlabel, ylabel):
    plt.figure(1)
    plt.title("Results")
    plt.xlabel(xlabel)  # 'Episode'
    plt.ylabel(ylabel)  # 'Total Reward'
    plt.plot([i for i in range(len(function_outputs))], function_outputs)


def create_one_hot(labels, num_classes):
    """
    One Hot Encoding of Labels

    From: [0 0 0 ... 1 1 1]
    To:
    [[1. 0.]
    [1. 0.]
    [1. 0.]
    ...
    [0. 1.]
    [0. 1.]
    [0. 1.]]
    """
    return np.eye(num_classes)[labels]



def custom_two_local(num_qubits):  # rx and linear cx
    circ = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        circ.rx(Parameter("theta_{}".format(i)), i)
    for i in range(num_qubits - 1):
        circ.cx(i, i + 1)

    return circ


random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

class VQCEnv(gym.Env):
    """
    State Space:
    n = number of qubits, d = maximum number of gates
    Example, number of qubits = 3, maximum depth = 4
    q0: [1, 0, 0, 0]
    q1: [0, 2, 0, 4]
    q2: [0, 0, 3, 5]
    Here, apply Rx on qubit q0. Then apply Ry on qubit q1. Then apply Rz on qubit q2.
    Then apply CNOT with control qubit q1 and target qubit q5

    Action Space (for the above scenario):
    0: Rx on qubit q0
    1: Rx on qubit q1
    2: Rx on qubit q2

    3: Ry on qubit q0
    4: Ry on qubit q1
    5: Ry on qubit q2

    6: Rz on qubit q0
    7: Rz on qubit q1
    8: Rz on qubit q2

    9: CNOT on (q0, q1)
    10: CNOT on (q0, q2)
    11: CNOT on (q1, q0)
    12: CNOT on (q1, q2)
    13: CNOT on (q2, q0)
    14: CNOT on (q2, q1)
    Size of Action Space: 3n + n(n - 1) + 1
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, vqc_experiment_data, rl_agent_data, render_mode="human"):
        super().__init__()

        """ VQC Experiment Parameters """
        self.df_train = vqc_experiment_data["df_train"].to_numpy()
        self.df_valid = vqc_experiment_data["df_valid"].to_numpy()
        self.df_test = vqc_experiment_data["df_test"].to_numpy()

        np.random.shuffle(self.df_train)

        self.valid_X = self.df_valid[:, 3:]
        self.valid_y = self.df_valid[:, 2].astype(int)
        self.test_X = self.df_test[:, 3:]
        self.test_y = self.df_test[:, 2].astype(int)

        self.valid_y = create_one_hot(self.valid_y, 2)
        self.test_y = create_one_hot(self.test_y, 2)

        print("Dataset Type: ", vqc_experiment_data["data_type"])
        print("Train Data Shape: ", self.df_train.shape)

        # sample training data (for VQC) in batches during the agent's training
        self.vqc_training_batch_size = vqc_experiment_data["vqc_training_batch_size"]

        # amplitude encoding
        if vqc_experiment_data["fmap"] == "amp":
            self.num_qubits = int(math.ceil(math.log2(vqc_experiment_data["dim"])))
        else:
            print("RL Agent Error: Only Amplitude Encoding is supported!")
            exit()

        self.random_seed = vqc_experiment_data["random_seed"]
        self.vqc_iterations = vqc_experiment_data["iterations"]
        self.reuse_params_flag = vqc_experiment_data["reuse_params_flag"]
        self.feature_map = RawFeatureVector(vqc_experiment_data["dim"])

        """ RL Agent Parameters """
        self.max_depth = rl_agent_data["max_depth"]

        # if True, then a barrier is added after every gate is added
        self.barrier_flag = rl_agent_data["barrier_flag"]

        self.print_intermediate_outputs = rl_agent_data["print_intermediate_outputs"]

        self.reward_function = rl_agent_data["reward_function"]

        # current reward system ('threshold'):
        # if f1 >= f1_upper_threshold then return reward of +5
        # if f1 < f1_lower_threshold then return reward of -5
        # else return reward of -0.1 (reduce gate depth)
        self.f1_upper_threshold = rl_agent_data["f1_upper_threshold"]
        self.f1_lower_threshold = rl_agent_data["f1_lower_threshold"]

        # current depth
        self.depth = 0
        self.reset_state = vqc_experiment_data["reset_state"]
        self.ansatz = QuantumCircuit(self.num_qubits)
        self.state = np.zeros(
            (self.num_qubits, self.max_depth)
        )  # no of qubits x max depth

        self.decode_cnot = {}
        counter = 0
        for i in range(self.num_qubits):
            for j in range(self.num_qubits):
                if i != j:
                    self.decode_cnot[counter] = (i, j)
                    counter += 1

        self.decode_gate = {1: "rx", 2: "ry", 3: "rz", 4: "cx", "none": "none"}


        # rotation gates + cnots + no action
        self.action_space = Discrete(
            30
        )

        self.observation_space = Box(
            low=0, high=5, shape=(self.num_qubits * self.max_depth,), dtype=np.intc
        )

        # theta_vec is the vector that contains the angles of all rotation gates
        # this vector gets optimized during VQC
        # you can have at most d gates
        self.param_vec = [Parameter(f"theta_{i}") for i in range(1000)]

        # number of parameters used
        self.num_params = 0

        # optimal values found in the latest VQC iteration
        self.param_values = []

        self.added_param_counter = 0

        self.latest_f1_score = 0
        self.best_f1_score = 0


        self.render_mode = render_mode

        # Default value is (-inf, inf)
        # self.reward_range = (lower bound, upper bound)



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        global prev_best_env
        
        self.state = np.zeros((self.num_qubits, self.max_depth))
        self.depth = 0
        self.ansatz = QuantumCircuit(
            self.num_qubits
        )  
        self.param_vec = [Parameter(f"theta_{i}") for i in range(1000)]
        self.num_params = 0
        self.param_values = []
        self.latest_f1_score = 0
        self.best_f1_score = 0
        self.added_param_counter = 0
        
        if self.reset_state == "prev_best":
            sample = np.random.rand()
            print("hiiiiiiiiiii",sample)
            
            if sample>0.5 and prev_best_env["depth"] < self.max_depth:
                self.state = deepcopy(prev_best_env["state"])
                self.depth = prev_best_env["depth"]

                for inst in prev_best_env["ansatz"].data:
                    if inst[0].name == "cx":
                        control = inst[1][0].index
                        target = inst[1][1].index
                        self.ansatz.cx(control,target)

                    elif inst[0].name == "rx":
                        param = inst[0].params[0]
                        qubit = inst[1][0].index
                        self.ansatz.rx(param,qubit)
                    
                    elif inst[0].name == "rz":
                        param = inst[0].params[0]
                        qubit = inst[1][0].index
                        self.ansatz.rz(param,qubit)

                    elif inst[0].name == "ry":
                        param = inst[0].params[0]
                        qubit = inst[1][0].index
                        self.ansatz.ry(param,qubit)  

                self.param_vec = [Parameter(f"theta_{i}") for i in range(1000)]
                self.num_params = prev_best_env["num_params"]
                self.param_values = deepcopy(prev_best_env["param_values"])
                self.latest_f1_score = prev_best_env["latest_f1_score"]
                self.best_f1_score = prev_best_env["best_f1_score"]
                self.added_param_counter = prev_best_env["added_param_counter"]
                

        elif self.reset_state == "heuristic":
            self.ansatz = custom_two_local(self.num_qubits)
            for i in range(self.num_qubits):
                self.state[0][i] = 1

            for i in range(self.num_qubits - 1):
                self.state[i][i + self.num_qubits] = 4
                self.state[i + 1][i + self.num_qubits] = 5
            self.depth = 2 * self.num_qubits - 1
            self.num_params = self.num_qubits
            self.param_values = np.random.random(self.num_qubits)
            self.added_param_counter = self.num_qubits
        
            f1 = self.test_ansatz_and_get_f1()
            self.latest_f1_score = f1
            self.best_f1_score = max(self.latest_f1_score, self.best_f1_score)


        # self.depth = 0
            
        
        state_np = np.array(self.state)
        state_tf = torch.from_numpy(state_np)
        flattened_state = state_tf.view(-1).to(device = device)

        return flattened_state, {"msg": "env got reset"}

    def decode_action(self, action):

        if self.depth==0:
            qubit = d1[action]
            if action<3:
                gate = 1
            elif action<6:
                gate = 2
            else:
                gate = 4

        elif self.depth ==1:
            qubit = d2[action]
            if action<3:
                gate = 2
            elif action<6:
                gate = 3
            else:
                gate = 4

        else:
            qubit = d3[action]
            if action<3:
                gate = 3
            elif action<6:
                gate = 1
            else:
                gate = 4
            

        if gate == 1:
            stats_buffer["RX_gate_added"][i_episode] += 1
        elif gate == 2:
            stats_buffer["RY_gate_added"][i_episode] += 1
        elif gate == 3:
            stats_buffer["RZ_gate_added"][i_episode] += 1
        else:
            stats_buffer["CX_gate_added"][i_episode] += 1
    
        return qubit, gate

    def step(self, action):  # needs changing (need to deal with parameters)random

        global random_flag
        global stats_buffer
        
        """
        The action lies in [0, 3n + 2 * nC2]
        Based on the action decoded, an appropriate gate is added
        to the state and the quantum circuit (ansatz)
        """

        step_start_time = time()

        if action!=None:
            qubit, gate= self.decode_action(action)
            print(qubit, self.decode_gate[gate])
            if type(self.param_values) is np.ndarray:
                self.param_values = self.param_values.tolist()
                
            if gate==1:
                self.ansatz.rx(self.param_vec[self.added_param_counter],qubit)
                self.state[qubit][self.depth] = gate
                self.num_params += 1
                self.added_param_counter += 1
            elif gate==2:
                self.ansatz.ry(self.param_vec[self.added_param_counter],qubit)
                self.state[qubit][self.depth] = gate
                self.num_params += 1
                self.added_param_counter += 1
            elif gate==3:
                self.ansatz.rz(self.param_vec[self.added_param_counter],qubit)
                self.state[qubit][self.depth] = gate
                self.num_params += 1
                self.added_param_counter += 1
            elif gate ==4:
                self.ansatz.cx(qubit[0],qubit[1])
                self.state[qubit[0]][self.depth] = gate
                self.state[qubit[1]][self.depth] = gate+1

        self.depth+=1
        print("\nIN STEP (rnd): ", self.param_values)


        logger.debug(
            "{} {} {}".format(
                "Random" if random_flag else "Greedy",
                self.decode_gate[gate],
                qubit)
        )

        print(
            "{} {} {}".format(
                "Random" if random_flag else "Greedy",
                self.decode_gate[gate],
                qubit)
        )



        f1_score = self.test_ansatz_and_get_f1()
        # depends on the reward function
        # reward = f1_score
        terminated = False

        if self.reward_function == "threshold":
            reward, terminated = self.threshold_reward_function(f1_score, type_='')
        elif self.reward_function == "direct":
            reward, terminated = self.direct_reward_function(f1_score, action)

        elif self.reward_function == "delta":
            reward, terminated = self.delta_reward(f1_score, action_type='single-qubit')
        else:
            print("Invalid Reward Function!")
            exit()

        self.latest_f1_score = f1_score

        self.best_f1_score = max(self.best_f1_score, self.latest_f1_score)

        episode_duration = time() - step_start_time

        if self.print_intermediate_outputs:
            print("Current Ansatz:")
            print(self.ansatz)

            print(
                f"Num Gates: {self.depth}\t f1 Score: {f1_score}\t Reward: {reward}\t Step Duration: {episode_duration}"
            )

        info = {"msg": "added gate"}

        # The state is flattened to 1D which makes it more convenient for the neural network
        state_np = np.array(self.state)
        state_tf = torch.from_numpy(state_np)
        flattened_state = state_tf.view(-1).to(device= device)

        print(flattened_state)
        print(flattened_state.shape)

        return flattened_state, reward, terminated, info

    def get_sampled_train_data(self, batch_size):
        """
        Returns a batch from the original train set that can then be used to
        train the VQC during each environment step. This way the agent sees
        most of the dataset, and each episode does not involve training
        over the entire dataset.
        """
        batch_indices = np.random.choice(self.df_train.shape[0], size=batch_size,replace = False)
        train_data = self.df_train[batch_indices]

        train_X = train_data[:, 3:]
        train_y = train_data[:, 2].astype(int)
        train_y = create_one_hot(train_y, 2)

        return train_X, train_y

    def delta_reward(self, f1_score, action_type):      #delta reward
        terminated = False
        if action_type == "single-qubit" or action_type == "two-qubit":
            reward = copy(f1_score)

        if self.depth>=250:
            terminated = True

        return reward, terminated

    def threshold_reward_function(self, f1_score, action_type):
        # if the weighted avg F1 score is above self.permitted_f1_threshold, then terminate
        if (
            f1_score >= self.f1_upper_threshold
        ):  # f1_upper_threshold, f1_lower_threshold
            terminated = True
            reward = 5
        elif f1_score < self.f1_lower_threshold or self.depth >= self.max_depth:
            terminated = True
            reward = -5
        elif action_type == "none":
            terminated = False
            reward = 0
        else:
            terminated = False
            reward = -0.1  # penalty for adding a gate

        return reward, terminated

    def direct_reward_function(self, f1_score, action):
        # trying f1 as soft score

        if self.depth > self.max_depth:
            terminated = True
            reward = -5
        else:
            terminated = False
            reward = f1_score

        # terminate when the agent chooses to place no gate
        if action == "none":
            terminated = True
            reward = 0

        return reward, terminated

    def test_ansatz_and_get_f1(self):
        """
        Test the given ansatz against a batch sample of the dataset
        Might take long!
        """

        print(self.ansatz.parameters)
        # Experimentally it turned out that reusing the parameters resulted in worse performance
        param_reuse_flag = False

        if param_reuse_flag:
            if not self.reuse_params_flag or self.num_params == 0:
                vqc = VQC(
                    feature_map=self.feature_map,
                    ansatz=self.ansatz,
                    optimizer=COBYLA(maxiter=10),
                    callback=callback_graph,
                    quantum_instance=QuantumInstance(
                        Aer.get_backend("qasm_simulator"),
                        shots=1000,
                        seed_simulator=self.random_seed,
                        seed_transpiler=self.random_seed,
                    ),
                )
            else:
                vqc = VQC(
                    feature_map=self.feature_map,
                    ansatz=self.ansatz,
                    optimizer=COBYLA(maxiter=10),
                    initial_point=self.param_values,
                    callback=callback_graph,
                    quantum_instance=QuantumInstance(
                        Aer.get_backend("qasm_simulator"),
                        shots=1000,
                        seed_simulator=self.random_seed,
                        seed_transpiler=self.random_seed,
                    ),
                )

        # do not reuse the parameters after each step
        else:
            vqc = VQC(
                feature_map=self.feature_map,
                ansatz=self.ansatz,
                optimizer=COBYLA(maxiter=10),
                callback=callback_graph,
                quantum_instance=QuantumInstance(
                    Aer.get_backend("qasm_simulator"),
                    shots=1000,
                    seed_simulator=self.random_seed,
                    seed_transpiler=self.random_seed,
                ),
            )

        print("PARAM VALS OLD: ", self.param_values)

        cur_iter = 0

        train_X, train_y = self.get_sampled_train_data(
            batch_size=self.vqc_training_batch_size
        )

        for iter_blk in range(1):
            # for iter_blk in range(3):  # 1 is not enough if you are reusing the parameters
            # for iter_blk in range((self.vqc_iterations-cur_iter)//10):  # original experiment

            vqc.fit(train_X, train_y)
            y_pred = vqc.predict(self.valid_X)
            final_pred = np.delete(y_pred, 1, 1)

            f1_score_weighted_avg = f1_score(
                self.valid_y.T[0], y_pred.T[0], average="weighted"
            )
            cur_iter += 10

            vqc.optimizer = COBYLA(maxiter=10)
            vqc.warm_start = True

            print("OPT PARAMS: ", vqc.initial_point, type(vqc.initial_point))

        self.param_values = vqc.initial_point

        print("PARAM VALS NEW: ", self.param_values)

        return f1_score_weighted_avg

    def render(self):
        pass

    def close(self):
        pass


"""
Load Datapoints
"""

dim = 64
data_type = "NvsBU"

df_train = pd.read_csv(
"./Data5/gnnfeatures_dim_64/features_bracs_hact_2_classes_pna_"
    + data_type
    + "_train.csv"

)
df_valid = pd.read_csv(
"./Data5/gnnfeatures_dim_64/features_bracs_hact_2_classes_pna_"
    + data_type
    + "_val.csv"
)
df_test = pd.read_csv(
"./Data5/gnnfeatures_dim_64/features_bracs_hact_2_classes_pna_"
    + data_type
    + "_test.csv"

)

# Add a zero column vector at the end (part of the original experiment, but not used)
## train_data_rf = np.hstack((train_data, np.expand_dims(np.zeros(train_data.shape[0]), axis = 1)))
## test_data_rf= np.hstack((test_data, np.expand_dims(np.zeros(test_data.shape[0]), axis = 1)))

# Parameters related to one VQC experiment

vqc_experiment_data = {
    "data_type": "NvsBU",
    "vqc_debug": False,
    "dim": 64,
    "fmap": "amp",
    "reps": 2,
    "dataset_size": 1000000,
    "random_seed": random_seed,
    "iterations": 50,
    "patience": 3,
    "vqc_training_batch_size": 200,
    "reuse_params_flag": True,
    "df_train": df_train,
    "df_valid": df_valid,
    "df_test": df_train,
    "reset_state": "empty" # (empty/heuristic/prev_best)
}

# Reward Function can be one of ['direct', 'threshold']
rl_agent_data = {
    "f1_upper_threshold": 0.65,
    "f1_lower_threshold": 0.4,
    "print_intermediate_outputs": True,
    "max_depth": 3
    ,  
    "barrier_flag": True,
    # 'reward_function': 'threshold',
    "reward_function": "delta",
    "use_pre_trained": False,
    'pre_trained_model_path': './trained/nn_model_full_9.pt'
}

# * Registering the environment
register(
    id="VQCEnv-v0",
    entry_point=f"{__name__}:VQCEnv",
)

def make_checkpoint(obj):
    # 162000: 45 hours
    with open("./checkpoints1/checkpoint_1.pkl", "wb") as f:
        pickle.dump(obj, f)

if os.path.exists("./checkpoints1/checkpoint_1.pkl"):
    with open("./checkpoints1/checkpoint_1.pkl", "rb") as f:
        checkpoint = pickle.load(f)

else:
    checkpoint = None

# Creating a gym environment
if checkpoint:
    env = checkpoint["env"]
    objective_values = checkpoint["objective_values"]
    prev_best_env = checkpoint['prev_env']
    num_states = 10**(2) + 10 + 1
    q_table = np.load('final_qtable.npy')

else:
    env = gym.make(
        "VQCEnv-v0",
        vqc_experiment_data=vqc_experiment_data,
        rl_agent_data=rl_agent_data,
    )
    objective_values = []
    prev_best_env = {
        "state": np.zeros((env.num_qubits, env.max_depth)),
        "depth": 0,
        "ansatz" :QuantumCircuit(
            env.num_qubits
        ),  
        "param_vec":[Parameter(f"theta_{i}") for i in range(1000)],
        "num_params":0,
        "param_values": [],
        "latest_f1_score": 0,
        "best_f1_score": 0,
        "added_param_counter": 0,
    }
    
    num_states = 10**(2) + 10 + 1
    q_table = np.zeros((num_states,10))

def callback_graph(_, objective_value):
    # clear_output(wait=True)
    objective_values.append(objective_value)



gamma = 0.99 #discount factor
lr = 0.1  #learning rate
eps_threshold = 0.1 

state_to_idx = {}
state_to_idx[tuple([0]*18)] = 0

counter=1
for i in range(10):
    state_rep = np.zeros((6,3))
    if i<3:
        state_rep[d1[i]][0] = 1
    elif i<6:
        state_rep[d1[i]][0] = 2
    else:
        state_rep[d1[i][0]][0] = 4
        state_rep[d1[i][1]][0] = 5
    
    state_to_idx[tuple(state_rep.flatten())]=counter
    counter+=1

for i in range(10):
    for j in range(10):
        state_rep = np.zeros((6,3))
        if i<3:
            state_rep[d1[i]][0] = 1
        elif i<6:
            state_rep[d1[i]][0] = 2
        else:
            state_rep[d1[i][0]][0] = 4
            state_rep[d1[i][1]][0] = 5

        if j<3:
            state_rep[d2[j]][1] = 2
        elif j<6:
            state_rep[d2[j]][1] = 3
        else:
            state_rep[d2[j][0]][1] = 4
            state_rep[d2[j][1]][1] = 5
        
        state_to_idx[tuple(state_rep.flatten())]=counter
        counter+=1


def select_action(state):
    global random_flag
    global steps_done
    global eps_threshold

    toss = np.random.rand()
    if toss>eps_threshold:
        action = torch.argmax(q_table).item()
        random_flag = False
    else:
        action = np.random.randint(0, n_actions)
        random_flag = True

    return action



def q_update(st,a,next_st,rew):  #perform

    st = tuple(st.squeeze(0).to(dtype = torch.int32).tolist())
    st = state_to_idx[st]

    next_st = tuple(next_st.squeeze(0).to(dtype = torch.int32).tolist())
    next_st = state_to_idx[next_st]

    if next_st is not None:
        q_table[st][a] = q_table+ lr*(gamma*max(q_table[next_st][a])+rew - q_table[st][a])
    else:
        q_table[st][a] = rew


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"********************** DEVICE: {device} **********************")
# Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

n_actions = env.action_space.n
state, info = env.reset()  # CHECK
n_observations = len(state)


wandb_run = wandb.init(
    # Set the project where this run will be logged
    name = "10_action_q_ucb",
    project="QRL1_3depth",
    entity="ep19b005",
    resume = True
)

random_flag = False
num_episodes = 10000

if checkpoint:
    stats_buffer = checkpoint["stats_buffer"]
    steps_done = checkpoint["steps_done"]
    ddqn_start_time = checkpoint["ddqn_start_time"]
    start = checkpoint["start"]
    eps_threshold = checkpoint["eps"]

else:
    stats_buffer = {
        "episodes": 0,
        "f1_scores": [[] for _ in range(num_episodes)],
        "ansatzes": [[] for _ in range(num_episodes)],
        "total_rewards": [[] for _ in range(num_episodes)],
        "QTable": [[] for _ in range(num_episodes)],
        "Ti(n)":[[0]*10 for _ in range(111)],
        "episode_durations": [],
        "steps_per_episode": [],
        "n_observations": n_observations,
        "n_actions": n_actions,
        "rand_greed_flag": [],
        "qubit": [],
        "gate": [], 
        "Total_number_of_gates_added": [0] * num_episodes,
        "RX_gate_added": [0] * num_episodes,
        "RY_gate_added": [0] * num_episodes,
        "RZ_gate_added": [0] * num_episodes,
        "CX_gate_added": [0] * num_episodes,
    }
    steps_done = 0 
    ddqn_start_time = time()
    start= 0 
    eps_threshold = 0.1

limit_flag = False

for i_episode in range(start, num_episodes):

    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    
    stats_buffer["f1_scores"][i_episode].append(env.latest_f1_score)
    stats_buffer["ansatzes"][i_episode].append(env.state.tolist())
    stats_buffer["total_rewards"][i_episode].append(total_reward)

    episode_start_time = time()
    logger.debug("START of Episode_{}".format(i_episode + 1))

    t=0

    max_f1_score = prev_best_env['latest_f1_score']
    ep_best_f1_score = env.latest_f1_score

    while True:
        if env.depth<3:
            action = select_action(state)
            observation, reward, terminated, info = env.step(action)
        else:
            observation, reward, terminated, info = env.step(None)


        print(f"Episode Num: {i_episode}\n\n\n")

        total_reward += reward

        reward = torch.tensor([reward], device=device)
        done = terminated

       
        stats_buffer["f1_scores"][i_episode].append(env.latest_f1_score)
        stats_buffer["ansatzes"][i_episode].append(env.state.tolist())
        stats_buffer["total_rewards"][i_episode].append(total_reward)


        logger.debug("Reward:{},Cumulative_Reward:{}".format(reward.item(), total_reward))
        logger.debug("F1_score:{}, Best F1_score (episode):{}\n".format(env.latest_f1_score,ep_best_f1_score))

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)

        # Q-update
        q_update(state,action,next_state,reward)

        state = next_state

        if done:
            print("Total Reward: ", total_reward)
            print("\n...end of episode")
            stats_buffer["Total_number_of_gates_added"][i_episode] = (
                stats_buffer["RX_gate_added"][i_episode]
                + stats_buffer["RY_gate_added"][i_episode]
                + stats_buffer["RZ_gate_added"][i_episode]
                + stats_buffer["CX_gate_added"][i_episode]
            )

            wandb.log({"Total_Reward": total_reward,
                       "F1_score": env.latest_f1_score,
                       "F1_score(e.o.e)": env.latest_f1_score,
                       "F1_score(episode_best)": ep_best_f1_score,
                       "Total_Reward(e.o.e)": total_reward,
                       "Total_steps_per_episode": t, 
                       "Addition steps": stats_buffer["Total_number_of_gates_added"][i_episode]})

            break

        else:
            wandb.log({"Total_Reward": total_reward,
                       "F1_score": env.latest_f1_score,
                       "F1_score(e.o.e)": 0,
                       "F1_score(episode_best)": 0,
                       "Total_Reward(e.o.e)": 0,
                       "Total_steps_per_episode": 0, 
                       "Addition steps": 0})

    logger.debug("END of Episode_{}".format(i_episode + 1))
    logger.debug("------------Epidsode Summary-------------------")
    logger.debug("Total Gates Added:{}".format(stats_buffer["Total_number_of_gates_added"][i_episode]))
    logger.debug("RX Gates Added:{}".format(stats_buffer["RX_gate_added"][i_episode]))
    logger.debug("RY Gates Added:{}".format(stats_buffer["RY_gate_added"][i_episode]))
    logger.debug("RZ Gates Added:{}".format(stats_buffer["RZ_gate_added"][i_episode]))
    logger.debug("CZ Gates Added:{}\n".format(stats_buffer["CX_gate_added"][i_episode]))

    logger.debug("--------------------------------------------------------")


    stats_buffer["episode_durations"].append(time() - episode_start_time)
    stats_buffer["episodes"] += 1
    stats_buffer["steps_per_episode"].append(t)

    filepath = "./json_objs5/exp_1.json"
    json_object = json.dumps(stats_buffer)
    with open(filepath, "w") as outfile:
        outfile.write(json_object)


    if time() - code_start > 162000:  # 162000(115 hrs)
        checkpoint = {
            "start": i_episode + 1,
            "env": env,
            "stats_buffer": stats_buffer,
            "steps_done": steps_done,
            "ddqn_start_time": ddqn_start_time,
            "objective_values": objective_values,
            "eps": eps_threshold,
            "prev_env" : prev_best_env
        }

        make_checkpoint(checkpoint)
        limit_flag = True
        break

if not limit_flag:
    checkpoint = {
            "start": i_episode + 1,
            "env": env,
            "stats_buffer": stats_buffer,
            "steps_done": steps_done,
            "ddqn_start_time": ddqn_start_time,
            "objective_values": objective_values,
            "eps": eps_threshold,
        }
    make_checkpoint(checkpoint)
    wandb.finish()
    

filepath = "./json_objs5/exp_1.json"
json_object = json.dumps(stats_buffer)
with open(filepath, "w") as outfile:
    outfile.write(json_object)

with open('final_qtable.npy', 'wb') as f:
    np.save(f, q_table)

print(
    f"DDQN Total Time for {num_episodes} episodes is: {time() - ddqn_start_time}"
)

