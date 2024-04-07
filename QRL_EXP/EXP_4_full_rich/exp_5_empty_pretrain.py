import sys
from copy import copy,deepcopy

import logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler("./logs5/exp_1.log", "a")
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
            6 * self.num_qubits + 2 * self.num_qubits * (self.num_qubits - 1) + 3
        )

        self.observation_space = Box(
            low=0, high=5, shape=(self.num_qubits * self.max_depth + 1,), dtype=np.intc
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

        self.prev_action = {i:float("inf") for i in range(2 * self.num_qubits)}

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
        self.prev_action = {i:float("inf") for i in range(2 * self.num_qubits)}
        
        if self.reset_state == "prev_best":
            sample = np.random.rand()
            print("hiiiiiiiiiii",sample)
            
            if sample>0.5 and prev_best_env["depth"] < self.max_depth:
                self.state = deepcopy(prev_best_env["state"])
                self.depth = prev_best_env["depth"]
                self.ansatz = deepcopy(prev_best_env["ansatz"])
                self.param_vec = [Parameter(f"theta_{i}") for i in range(1000)]
                self.num_params = prev_best_env["num_params"]
                self.param_values = deepcopy(prev_best_env["param_values"])
                self.latest_f1_score = prev_best_env["latest_f1_score"]
                self.best_f1_score = prev_best_env["best_f1_score"]
                self.added_param_counter = prev_best_env["added_param_counter"]
                self.prev_action = deepcopy(prev_best_env["prev_action"])
                
            print(self.state)
            print(self.depth)
            print(self.ansatz) 
            print(self.num_params)
            print(self.param_values)
            print(self.latest_f1_score)
            print(self.best_f1_score)
            print(self.added_param_counter)
        
        
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

            for k,v in self.decode_cnot.items():
                for l in range(self.num_qubits-1):
                    if v==(l,l+1):
                        self.prev_action[self.num_qubits + v[1]]= k
            
            for i in range(self.num_qubits):
                self.prev_action[i] = 3 * i
                if i==0:
                    self.prev_action[i+self.num_qubits] = 3*i + (3 * self.num_qubits)

        # self.depth = 0
            
        
        state_np = np.array(self.state)
        state_tf = torch.from_numpy(state_np)
        flattened_state = torch.cat(
            (state_tf.view(-1), torch.tensor([self.best_f1_score]))
        ).to(device=device)

        return flattened_state, {"msg": "env got reset"}

    def decode_action(self, action):

        if action == self.action_space.n - 3:
            type_ = "none"
            qubit = -1
            gate = "none"
            pos = "none"


        elif action < 6 * self.num_qubits:
            type_ = "single-qubit"
            pos = action // (3 * self.num_qubits)
            qubit = (action % (3 * self.num_qubits)) // 3
            gate = (action % (3 * self.num_qubits)) % (3) + 1

            if gate == 1:
                stats_buffer["RX_gate_added"][i_episode] += 1
            elif gate == 2:
                stats_buffer["RY_gate_added"][i_episode] += 1
            elif gate == 3:
                stats_buffer["RZ_gate_added"][i_episode] += 1

            if self.prev_action[self.num_qubits*pos + qubit] == float("inf"):

                self.prev_action[self.num_qubits + qubit] =  action if not pos else (action% (3*self.num_qubits))
                self.prev_action[qubit] = action if pos else action+(3*self.num_qubits)
            
            else:
                self.prev_action[self.num_qubits * pos + qubit] = action

            

        elif action < self.action_space.n - 3:
            type_ = "two-qubit"
            pos = (action - 6 * self.num_qubits) // (
                self.num_qubits * (self.num_qubits - 1)
            )
            a = (action - 6 * self.num_qubits) % (
                self.num_qubits * (self.num_qubits - 1)
            )
            qubit = self.decode_cnot[a]
            gate = 4

            stats_buffer["CX_gate_added"][i_episode] += 1

            if self.prev_action[self.num_qubits*pos + qubit[1]] == float("inf"):
                a = action - 6*self.num_qubits
                self.prev_action[self.num_qubits + qubit[1]] =  action if not pos else (6*self.num_qubits + (a % ((self.num_qubits-1)*self.num_qubits)))
                self.prev_action[qubit[1]] = action if pos else action+(self.num_qubits*(self.num_qubits-1))
            
            else:
                self.prev_action[self.num_qubits * pos + qubit[1]] = action 
            


        elif action == self.action_space.n - 2:
            type_ = "removal"
            pos = 1
            qubit = -1
            gate = "none"

        else:
            type_ = "removal"
            gate = "none"
            qubit = -1
            pos = 0
        return type_, qubit, gate, pos

    def step(self, action):  # needs changing (need to deal with parameters)random

        global random_flag
        
        """
        The action lies in [0, 3n + 2 * nC2]
        Based on the action decoded, an appropriate gate is added
        to the state and the quantum circuit (ansatz)
        """

        step_start_time = time()

        type_, qubit, gate, pos = self.decode_action(action)
        print(type_, qubit, gate, pos)


        if type(self.param_values) is np.ndarray:
            self.param_values = self.param_values.tolist()

        if type_ == "single-qubit":
            if pos == 0:
                tmp = np.zeros((self.num_qubits, 1))
                tmp[qubit][0] = gate
                self.state = np.hstack(
                    (tmp, self.state[:, :-1])
                )  # remove the last column completely...
                self.param_values.insert(0, np.random.rand())
                self.ansatz.data.insert(
                    0,
                    CircuitInstruction(
                        operation=Instruction(
                            name=self.decode_gate[gate],
                            num_qubits=1,
                            num_clbits=0,
                            params=[self.param_vec[self.added_param_counter]],
                        ),
                        qubits=(Qubit(QuantumRegister(self.num_qubits, "q"), qubit),),
                        clbits=(),
                    ),
                )

            else:
                self.state[qubit][self.depth] = gate
                self.param_values.append(np.random.rand())
                self.ansatz.data.append(
                    CircuitInstruction(
                        operation=Instruction(
                            name=self.decode_gate[gate],
                            num_qubits=1,
                            num_clbits=0,
                            params=[self.param_vec[self.added_param_counter]],
                        ),
                        qubits=(Qubit(QuantumRegister(self.num_qubits, "q"), qubit),),
                        clbits=(),
                    )
                )

            print("\nIN STEP (rnd): ", self.param_values)
            self.depth += 1
            self.num_params += 1
            self.added_param_counter += 1

        elif type_ == "two-qubit":
            if pos == 0:
                tmp = np.zeros((self.num_qubits, 1))
                tmp[qubit[0]][0] = 4
                tmp[qubit[1]][0] = 5
                self.state = np.hstack((tmp, self.state[:, :-1]))
                self.ansatz.data.insert(
                    0,
                    CircuitInstruction(
                        operation=Instruction(
                            name="cx", num_qubits=2, num_clbits=0, params=[]
                        ),
                        qubits=(
                            Qubit(QuantumRegister(self.num_qubits, "q"), qubit[0]),
                            Qubit(QuantumRegister(self.num_qubits, "q"), qubit[1]),
                        ),
                        clbits=(),
                    ),
                )

            else:
                self.state[qubit[0]][self.depth] = 4
                self.state[qubit[1]][self.depth] = 5

                self.ansatz.data.append(
                    CircuitInstruction(
                        operation=Instruction(
                            name="cx", num_qubits=2, num_clbits=0, params=[]
                        ),
                        qubits=(
                            Qubit(QuantumRegister(self.num_qubits, "q"), qubit[0]),
                            Qubit(QuantumRegister(self.num_qubits, "q"), qubit[1]),
                        ),
                        clbits=(),
                    )
                )

            self.depth += 1

        elif type_ == "removal":
            if self.ansatz.data[-pos][0].name != "cx":
                self.num_params -= 1
                self.param_values.pop(-pos)

            if pos == 0:
                tmp = np.zeros((self.num_qubits, 1))
                self.state = np.hstack((self.state[:, 1:], tmp))
                removed = self.ansatz.data[0]
                self.ansatz.data = self.ansatz.data[1:]

            else:
                self.state[:, self.depth - 1] = np.zeros(self.num_qubits)
                removed = self.ansatz.data[-1]
                self.ansatz.data = self.ansatz.data[:-1]

            if removed[0].name == "rx":
                stats_buffer["RX_gate_removed"][i_episode] += 1
                gate = 1
            elif removed[0].name == "ry":
                stats_buffer["RY_gate_removed"][i_episode] += 1
                gate = 2
            elif removed[0].name == "rz":
                stats_buffer["RZ_gate_removed"][i_episode] += 1
                gate = 3
            else:
                stats_buffer["CX_gate_removed"][i_episode] += 1
                gate = 4

            self.depth -= 1

        elif type_ == "none":
            # this is needed in case the agent is happy with the current circuit,
            # it will learn to stop adding gates

            pass
        
        else:
            print("error in step(): invalid action. ", type_, qubit, gate, action)
            exit()

        logger.debug(
            "{} {} {} {} {}".format(
                "Random" if random_flag else "Greedy",
                type_,
                self.decode_gate[gate],
                qubit,
                "Right" if pos else "Left",
            )
        )

        f1_score = self.test_ansatz_and_get_f1()
        # depends on the reward function
        # reward = f1_score
        terminated = False

        if self.reward_function == "threshold":
            reward, terminated = self.threshold_reward_function(f1_score, type_)
        elif self.reward_function == "direct":
            reward, terminated = self.direct_reward_function(f1_score, action)
        elif self.reward_function == "delta":
            reward, terminated = self.delta_reward(f1_score, type_)
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
        flattened_state = torch.cat(
            (state_tf.view(-1), torch.tensor([self.best_f1_score]))
        ).to(device=device)

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
        batch_indices = np.random.choice(self.df_train.shape[0], size=batch_size)
        train_data = self.df_train[batch_indices]

        train_X = train_data[:, 3:]
        train_y = train_data[:, 2].astype(int)
        train_y = create_one_hot(train_y, 2)

        return train_X, train_y

    def delta_reward(self, f1_score, action_type):
        terminated = False
        if action_type == "single-qubit" or action_type == "two-qubit":
            reward = (-0.01 + (f1_score - self.best_f1_score)) if self.depth != 1 else 0
        elif action_type == "removal":
            reward = (0.01 + (f1_score - self.best_f1_score)) if self.depth != 0 else 0
        elif action_type == "none":
            reward = 0

        if self.depth >= self.max_depth or action_type == 'none':
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
    "reset_state": "prev_best" # (empty/heuristic/prev_best)
}

# Reward Function can be one of ['direct', 'threshold']
rl_agent_data = {
    "f1_upper_threshold": 0.65,
    "f1_lower_threshold": 0.4,
    "print_intermediate_outputs": True,
    "max_depth": 20
    ,  
    "barrier_flag": True,
    # 'reward_function': 'threshold',
    "reward_function": "delta",
    "use_pre_trained": True,
    'pre_trained_model_path': './trained/nn_model_7.pt'
}

# * Registering the environment
register(
    id="VQCEnv-v0",
    entry_point=f"{__name__}:VQCEnv",
)


class ReplayMemory(object):
    """
    Memory buffer that stores the episodes encountered
    Improves stability
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        """
        Simple fully connected neural network with 3 layers
        This can be experimented upon
        """
        super(DQN, self).__init__()
        print("n_obs: ", n_observations)
        print("n_act: ", n_actions)
        self.layer1 = nn.Linear(n_observations, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512, 512)
        self.layer5 = nn.Linear(512, 512)
        self.layer6 = nn.Linear(512, 512)
        self.layer7 = nn.Linear(512, 512)
        self.layer8 = nn.Linear(512, 512)
        self.layer9 = nn.Linear(512,n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = F.relu(self.layer8(x))
        x = self.layer9(x)

        return x


def make_checkpoint(obj):
    # 162000: 45 hours
    with open("./checkpoints1/checkpoint_1.pkl", "wb") as f:
        pickle.dump(obj, f)

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))




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
        "prev_action": {i:float("inf") for i in range(2 * env.num_qubits)}
    }




# * Creating a Transition object


# * Objective value: used in callback_graph


def callback_graph(_, objective_value):
    # clear_output(wait=True)
    objective_values.append(objective_value)


# class RandomAgent:
#     """
#     Simple agent that samples an action uniformly for each step
#     Useful for debugging the environment
#     """

#     def __init__(self, env):
#         self.env = env

#     def select_action(self, observation):
#         """
#         observation - current state
#         """
#         return self.env.action_space.sample()


# """
# Test with Random Agent
# """

# TEST_RANDOM_AGENT_FLAG = False

# if TEST_RANDOM_AGENT_FLAG:
#     random_agent = RandomAgent(env)
#     num_eps = 2
#     return_over_all_eps = []

#     for episode in range(num_eps):
#         observation, info = env.reset()
#         total_reward = 0

#         while True:
#             action = random_agent.select_action(observation)
#             observation, reward, terminated, info = env.step(action)
#             total_reward += reward

#             if terminated:
#                 print("Total Reward: ", total_reward)
#                 return_over_all_eps.append(total_reward)
#                 print("\n...end of episode")
#                 break

#     avg_return_random_agent = np.mean(return_over_all_eps)
#     plot_line_graph(return_over_all_eps, "Episode", "Total Reward")
#     print("Average Total Return: ", np.mean(return_over_all_eps))
"""
DDQN Agent
"""

BATCH_SIZE = 32
GAMMA = 0.99
TAU = 0.005  # for soft update of target network
LR = 1e-4  # learning rate for adam optimizer
# BUFFER_CAPACITY = 10000
BUFFER_CAPACITY = 10000  # keep small buffer size so that training starts early


def select_action(state):

    global random_flag
    global steps_done
    global eps_threshold


    action_ch = list(range(n_actions-2))
    action_mask = torch.zeros(n_actions, device=device).unsqueeze(0)

    sample = random.random()
    eps_threshold = (0.5 * np.exp(-steps_done/62.15)) if steps_done<=100 else 0.1 
    
    steps_done += 1

    if sample>=eps_threshold:
        random_flag = False
        for _,v in env.prev_action.items():
            if v != float("inf"):
                action_mask[0][v] = -float("inf")
        
        if env.depth <= 1 or (env.num_params==1 and (env.ansatz.data[0][0].name!='cx' or env.ansatz.data[-1][0].name!='cx')):
            action_mask[0][-1] = -float("inf")
            action_mask[0][-2] = -float("inf")
            
        if env.depth == 0:
            action_mask[0][-3] = -float("inf")

    # Using the epsilon greedy approach with a high epsilon value in the beginning to encourage exploration
        with torch.no_grad():
            tmp = policy_net(state) + action_mask
            return tmp.max(1)[1].view(1, 1)
        
    else:
        # explore valid actions for the current state
        random_flag = True
        add_remove_threshold = random.random()
        
        if env.depth<=1 or (env.num_params==1 and (env.ansatz.data[0][0].name!='cx' or env.ansatz.data[-1][0].name!='cx')):
            add_remove_threshold =0.5 
            if env.depth==0:
                action_ch.remove(n_actions-3)


        if add_remove_threshold >= 0.5:
            for _,v in env.prev_action.items():
                if v!= float("inf"):
                    if v not in action_ch:
                        continue
                    else:
                        action_ch.remove(v)
            return torch.tensor([[np.random.choice(action_ch)]],device= device, dtype = torch.long)
        
        else:
            return torch.tensor([[np.random.choice([n_actions-1,n_actions-2])]], device= device, dtype = torch.long)


def optimize_model():  # dealing with the loss_function (linear output)-(changing action space) -(i think not needed only valid actions
    # lie in the Replay Memory)
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device = device)
    state_batch = torch.cat(batch.state).to(device=device)
    action_batch = torch.cat(batch.action).to(device=device)
    reward_batch = torch.cat(batch.reward).to(device=device)

    # action taken for each batch state
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # compute the Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)

    optimizer.step()


def save_experiment_models(
    policy_net, target_net, memory, experiment_number, print_outputs=False
):
    print("\n--- saving experiment data ---\n")

    torch.save(policy_net.state_dict(), f"./models/policy_net_{experiment_number}.pt")
    torch.save(target_net.state_dict(), f"./models/target_net_{experiment_number}.pt")

    file_object = open(f"./models/memory_{experiment_number}.obj", "wb")
    pickle.dump(memory, file_object)
    file_object.close()

    # Test whether the data was saved correctly
    if print_outputs:
        print("\n\nModel Parameters")
        for param in policy_net.parameters():
            print(param.data)

        print("memory", memory.memory)
        print("done\n\n")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"********************** DEVICE: {device} **********************")
# Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

n_actions = env.action_space.n
state, info = env.reset()  # CHECK
n_observations = len(state)



policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)

if rl_agent_data['use_pre_trained']:


    a = torch.load(rl_agent_data["pre_trained_model_path"],map_location=device)   # pre trained
    minimum = 1/(policy_net.layer9.in_features**0.5)
    layer_9_w = ((torch.rand((n_actions,policy_net.layer9.in_features)) * (2* minimum))-minimum)
    layer_9_b = (torch.rand(n_actions)* (2*minimum) - minimum)

    del a['f1_layer.bias']
    del a['f1_layer.weight']

    a['layer9.weight'] = layer_9_w
    a['layer9.bias'] = layer_9_b

    policy_net.load_state_dict(a)


wandb_run = wandb.init(
    # Set the project where this run will be logged
    name = "exp_4_prev_best_pretrain_full_rich",
    project="QRL1_test",
    entity="ep19b005",
    resume=True,
)


random_flag = False
num_episodes = 1000

if checkpoint:
    stats_buffer = checkpoint["stats_buffer"]
    policy_net.load_state_dict(checkpoint["policy_net"])
    target_net.load_state_dict(checkpoint["target_net"])
    memory = checkpoint["memory"]
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
        "episode_durations": [],
        "steps_per_episode": [],
        "n_observations": n_observations,
        "n_actions": n_actions,
        "Total_number_of_gates_added": [0] * num_episodes,
        "Total_number_of_gates_removed": [0] * num_episodes,
        "RX_gate_added": [0] * num_episodes,
        "RY_gate_added": [0] * num_episodes,
        "RZ_gate_added": [0] * num_episodes,
        "CX_gate_added": [0] * num_episodes,
        "RX_gate_removed": [0] * num_episodes,
        "RY_gate_removed": [0] * num_episodes,
        "RZ_gate_removed": [0] * num_episodes,
        "CX_gate_removed": [0] * num_episodes
    }
    memory = ReplayMemory(BUFFER_CAPACITY)
    target_net.load_state_dict(policy_net.state_dict())
    steps_done = 0
    ddqn_start_time = time()
    start=0
    eps_threshold = 0.5  


optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)


steps_done = 0

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
        print(prev_best_env["ansatz"].data)
        print(prev_best_env["state"])
        print(prev_best_env["depth"])
        print(prev_best_env["ansatz"]) 
        print(prev_best_env["num_params"])
        print(prev_best_env["param_values"])
        print(prev_best_env["latest_f1_score"])
        print(prev_best_env["best_f1_score"])
        print(prev_best_env["added_param_counter"])
        
        action = select_action(state)
        observation, reward, terminated, info = env.step(action.item())


        print(f"Episode Num: {i_episode}\n\n\n")

        total_reward += reward

        logger.debug("Reward:{},Cumulative_Reward:{}".format(reward, total_reward))
        logger.debug("F1_score:{}, Best F1_score (episode):{}\n".format(env.latest_f1_score,ep_best_f1_score))

        reward = torch.tensor([reward], device=device)
        done = terminated

        
        wandb.log({"Total_Reward": total_reward})
        wandb.log({"F1_score": env.latest_f1_score})
        

        stats_buffer["f1_scores"][i_episode].append(env.latest_f1_score)
        stats_buffer["ansatzes"][i_episode].append(env.state.tolist())
        stats_buffer["total_rewards"][i_episode].append(total_reward)

        if env.latest_f1_score > max_f1_score and env.reset_state=="prev_best":
            
            prev_best_env["state"]  = deepcopy(env.state)
            prev_best_env["depth"]  = env.depth
            prev_best_env["ansatz"] = deepcopy(env.ansatz)
            prev_best_env["num_params"] = env.num_params
            prev_best_env["param_values"] = deepcopy(env.param_values)
            prev_best_env["latest_f1_score"] = env.latest_f1_score
            prev_best_env["best_f1_score"] = env.best_f1_score
            prev_best_env["added_param_counter"] = env.added_param_counter
            prev_best_env["prev_action"] = deepcopy(env.prev_action)
            
            max_f1_score = env.latest_f1_score
            print("hi")
            print(prev_best_env["ansatz"])
        
        if env.latest_f1_score > ep_best_f1_score:
            ep_best_f1_score = env.latest_f1_score

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        # soft update of target network. TAU is small and the update is like the weighted mean of policy and target network
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)
        t += 1

        if done:
            print("Total Reward: ", total_reward)
            print("\n...end of episode")
            stats_buffer["Total_number_of_gates_added"][i_episode] = (
                stats_buffer["RX_gate_added"][i_episode]
                + stats_buffer["RY_gate_added"][i_episode]
                + stats_buffer["RZ_gate_added"][i_episode]
                + stats_buffer["CX_gate_added"][i_episode]
            )
            stats_buffer["Total_number_of_gates_removed"][i_episode] = (
                stats_buffer["RX_gate_removed"][i_episode]
                + stats_buffer["RY_gate_removed"][i_episode]
                + stats_buffer["RZ_gate_removed"][i_episode]
                + stats_buffer["CX_gate_removed"][i_episode]
            )
            break

    logger.debug("END of Episode_{}".format(i_episode + 1))
    logger.debug("------------Epidsode Summary-------------------")
    logger.debug("Total Gates Added:{}".format(stats_buffer["Total_number_of_gates_added"][i_episode]))
    logger.debug("RX Gates Added:{}".format(stats_buffer["RX_gate_added"][i_episode]))
    logger.debug("RY Gates Added:{}".format(stats_buffer["RY_gate_added"][i_episode]))
    logger.debug("RZ Gates Added:{}".format(stats_buffer["RZ_gate_added"][i_episode]))
    logger.debug("CZ Gates Added:{}\n".format(stats_buffer["CX_gate_added"][i_episode]))

    logger.debug("Total Gates Removed:{}".format(stats_buffer["Total_number_of_gates_removed"][i_episode]))
    logger.debug("RX Gates Removed:{}".format(stats_buffer["RX_gate_removed"][i_episode]))
    logger.debug("RY Gates Removed:{}".format(stats_buffer["RY_gate_removed"][i_episode]))
    logger.debug("RZ Gates Removed:{}".format(stats_buffer["RZ_gate_removed"][i_episode]))
    logger.debug("CZ Gates Removed:{}\n".format(stats_buffer["CX_gate_removed"][i_episode]))

    logger.debug("--------------------------------------------------------")

    wandb.log({"F1_score(e.o.e)": env.latest_f1_score})
    wandb.log({"F1_score(episode_best)": ep_best_f1_score})
    wandb.log({"Total_Reward(e.o.e)": total_reward})
    

    stats_buffer["episode_durations"].append(time() - episode_start_time)
    stats_buffer["episodes"] += 1
    stats_buffer["steps_per_episode"].append(t)

    wandb.log({"Total_steps_per_episode": t, "Addition steps": stats_buffer[
        "Total_number_of_gates_added"][i_episode], "Removal steps": stats_buffer[
            "Total_number_of_gates_removed"][i_episode]})
    

    if time() - code_start > 162000:  # 162000
        checkpoint = {
            "start": i_episode + 1,
            "env": env,
            "stats_buffer": stats_buffer,
            "target_net": target_net.state_dict(),
            "policy_net": policy_net.state_dict(),
            "steps_done": steps_done,
            "ddqn_start_time": ddqn_start_time,
            "memory": memory,
            "objective_values": objective_values,
            "eps": eps_threshold,
            "prev_env" : prev_best_env
        }

        make_checkpoint(checkpoint)
        limit_flag = True
        break

if not limit_flag:
    make_checkpoint(checkpoint)
    
    wandb.finish()
    

filepath = "./json_objs5/exp_1.json"
json_object = json.dumps(stats_buffer)
with open(filepath, "w") as outfile:
    outfile.write(json_object)

print(
    f"DDQN Total Time for {num_episodes} episodes is: {time() - ddqn_start_time}"
)

