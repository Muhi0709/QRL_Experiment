# %% [code]
# %% [code]
# %% [code]
# %% [code]
import sys
import subprocess
from copy import copy,deepcopy

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'qiskit==0.43.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'qiskit-machine-learning==0.6.1'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gym==0.26'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pylatexenc'])


import wandb


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from time import time
import json

from sklearn.metrics import f1_score

import logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler("./logs/log.log", "a")
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

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

import pandas as pd
import numpy as np

import random

random_seed = 12345678
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

start_time = time()

wandb.login(key="63cea33f1c9c226a7fdeffedfc683b2b4398cfb5")

wandb_run = wandb.init(
    # Set the project where this run will be logged
    name = "spsa_no_pretrain_lr=0.001",
    project="QRL_spsa",
    entity="ep19b005",
    resume=True,
)

def create_one_hot(labels, num_classes):
    
    return np.eye(num_classes)[labels]

objective_values = []
def callback_graph(_, objective_value):
    # clear_output(wait=True)
    objective_values.append(objective_value)

data_type = "NvsBU"
df_train = pd.read_csv(
"./Data/gnnfeatures_dim_64/features_bracs_hact_2_classes_pna_"
    + data_type
    + "_train.csv"
).to_numpy()
df_valid = pd.read_csv(
"./Data/gnnfeatures_dim_64/features_bracs_hact_2_classes_pna_"
    + data_type
    + "_val.csv"
).to_numpy()
df_test = pd.read_csv(
"./Data/gnnfeatures_dim_64/features_bracs_hact_2_classes_pna_"
    + data_type
    + "_test.csv"

).to_numpy()

np.random.shuffle(df_train)
valid_X = df_valid[:, 3:]
valid_y = df_valid[:, 2].astype(int)
valid_y = create_one_hot(valid_y, 2)

def get_sampled_train_data(df_train,batch_size=200):

    batch_indices = np.random.choice(df_train.shape[0], size=batch_size)
    train_data = df_train[batch_indices]

    train_X = train_data[:, 3:]
    train_y = train_data[:, 2].astype(int)
    train_y = create_one_hot(train_y, 2)

    return train_X, train_y

num_qubits = 6
max_depth = 20
n_actions = 6* num_qubits + 2*num_qubits*(num_qubits-1) + 3

decode_cnot = {}
counter = 0
for i in range(num_qubits):
    for j in range(num_qubits):
        if i != j:
            decode_cnot[counter] = (i, j)
            counter += 1

decode_gate = {1: "rx", 2: "ry", 3: "rz", 4: "cx", "none": "none"}

class DQN(nn.Module):
    def __init__(self, n_observations,n_actions):

        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(256, 256)
        self.layer6 = nn.Linear(256,n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = self.layer6(x)

        return x


def select_action(dqn,state,eps_threshold,depth,
                  num_params,ansatz,prev_action):

    global random_flag
    
    state = torch.tensor(state,dtype=torch.float32).unsqueeze(0)
    
    action_mask = torch.zeros(n_actions).unsqueeze(0)


    for _,v in prev_action.items():
        if v != float("inf"):
            action_mask[0][v] = -float("inf")
        
    if depth <= 1 or (num_params==1 and (ansatz.data[0][0].name!='cx' or ansatz.data[-1][0].name!='cx')):
        action_mask[0][-1] = -float("inf")
        action_mask[0][-2] = -float("inf")
            
    if depth == 0:
        action_mask[0][-3] = -float("inf")

    # Using the epsilon greedy approach with a high epsilon value in the beginning to encourage exploration
        with torch.no_grad():
            tmp = dqn(state) + action_mask   
            prob = nn.Softmax(dim=1)
            probabilities = prob(tmp).squeeze(0).tolist()
            return np.random.choice(n_actions,p = probabilities)  #running into sum not equal to 1 error(any work around?????)


def decode_action(action,prev_action):

    if action == n_actions - 3:
        type_ = "none"
        qubit = -1
        gate = "none"
        pos = "none"


    elif action < 6 * num_qubits:
        type_ = "single-qubit"
        pos = action // (3 * num_qubits)
        qubit = (action % (3 * num_qubits)) // 3
        gate = (action % (3 * num_qubits)) % (3) + 1

        if prev_action[num_qubits*pos + qubit] == float("inf"):

            prev_action[num_qubits + qubit] =  action if pos else (action+ 3*num_qubits)
            prev_action[qubit] = action if not pos else action-(3*num_qubits)
            
        else:
            prev_action[num_qubits * pos + qubit] = action

            

    elif action < n_actions - 3:
        type_ = "two-qubit"
        pos = (action - 6 * num_qubits) // (
                num_qubits * (num_qubits - 1)
            )
        a = (action - 6 * num_qubits) % (
        num_qubits * (num_qubits - 1)
            )
        qubit = decode_cnot[a]
        gate = 4


        if prev_action[num_qubits*pos + qubit[1]] == float("inf"):
            a = action - 6*num_qubits
            prev_action[num_qubits + qubit[1]] =  action if pos else action+(num_qubits*(num_qubits-1))
            prev_action[qubit[1]] = action if not pos else action-(num_qubits*(num_qubits-1))
            
        else:
            prev_action[num_qubits * pos + qubit[1]] = action 
            

    elif action == n_actions - 2:
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

def step(action, prev_action, param_values,ansatz,
         added_param_counter,param_vec,
         state,depth):  # needs changing (need to deal with parameters)random

    global random_flag
        

    type_, qubit, gate, pos = decode_action(action,prev_action)
    print(type_, qubit, gate, pos)
    
    if type(param_values) is np.ndarray:
        param_values = param_values.tolist()

    if type_ == "single-qubit":
        if pos == 0:
            tmp = np.zeros((num_qubits, 1))
            tmp[qubit][0] = gate
            state = np.hstack(
                    (tmp, state[:, :-1])
                )  # remove the last column completely...
            param_values.insert(0, np.random.rand())
            ansatz.data.insert(
                    0,
                    CircuitInstruction(
                        operation=Instruction(
                            name=decode_gate[gate],
                            num_qubits=1,
                            num_clbits=0,
                            params=[param_vec[added_param_counter]],
                        ),
                        qubits=(Qubit(QuantumRegister(num_qubits, "q"), qubit),),
                        clbits=(),
                    ),
                )

        else:
            state[qubit][depth] = gate
            param_values.append(np.random.rand())
            ansatz.data.append(
            CircuitInstruction(
                operation=Instruction(
                name=decode_gate[gate],
                num_qubits=1,
                num_clbits=0,
                params=[param_vec[added_param_counter]],
                ),
                qubits=(Qubit(QuantumRegister(num_qubits, "q"), qubit),),
                    clbits=(),
                )
            )

        depth += 1
        added_param_counter += 1

    elif type_ == "two-qubit":
        if pos == 0:
            tmp = np.zeros((num_qubits, 1))
            tmp[qubit[0]][0] = 4
            tmp[qubit[1]][0] = 5
            state = np.hstack((tmp, state[:, :-1]))
            ansatz.data.insert(
                    0,
                    CircuitInstruction(
                        operation=Instruction(
                            name="cx", num_qubits=2, num_clbits=0, params=[]
                        ),
                        qubits=(
                            Qubit(QuantumRegister(num_qubits, "q"), qubit[0]),
                            Qubit(QuantumRegister(num_qubits, "q"), qubit[1]),
                        ),
                        clbits=(),
                    ),
            )

        else:
            state[qubit[0]][depth] = 4
            state[qubit[1]][depth] = 5

            ansatz.data.append(
                    CircuitInstruction(
                        operation=Instruction(
                            name="cx", num_qubits=2, num_clbits=0, params=[]
                        ),
                        qubits=(
                            Qubit(QuantumRegister(num_qubits, "q"), qubit[0]),
                            Qubit(QuantumRegister(num_qubits, "q"), qubit[1]),
                        ),
                        clbits=(),
                    )
                )

        depth += 1

    elif type_ == "removal":
        if ansatz.data[-pos][0].name != "cx":
            param_values.pop(-pos)

        if pos == 0:
            tmp = np.zeros((num_qubits, 1))
            state = np.hstack((state[:, 1:], tmp))
            ansatz.data = ansatz.data[1:]

        else:
            state[:, depth - 1] = np.zeros(num_qubits)
            ansatz.data = ansatz.data[:-1]

        depth -= 1

    elif type_ == "none":
            # this is needed in case the agent is happy with the current circuit,
            # it will learn to stop adding gates

            pass
        
    else:
        print("error in step(): invalid action. ", type_, qubit, gate, action)
        exit()
    
    f1_score, param_values  = test_ansatz_and_get_f1(ansatz)
        # depends on the reward function
        # reward = f1_score
    print("No_of_gates:",depth)
    terminated = False
    if depth==max_depth or type_ == "none":
        terminated = True

     # The state is flattened to 1D which makes it more convenient for the neural network
        
    state_np = np.array(state)
    state_tf = torch.from_numpy(state_np)
    flattened_state = state_tf.view(-1)


    return (flattened_state, f1_score, terminated, param_values,ansatz,
            added_param_counter,state,depth)

def get_sampled_train_data(batch_size):

    batch_indices = np.random.choice(df_train.shape[0], size=batch_size)
    train_data = df_train[batch_indices]

    train_X = train_data[:, 3:]
    train_y = train_data[:, 2].astype(int)
    train_y = create_one_hot(train_y, 2)

    return train_X, train_y



def test_ansatz_and_get_f1(ansatz):

    vqc = VQC(
        feature_map=RawFeatureVector(64),
        ansatz= ansatz,
        optimizer=COBYLA(maxiter=10),
        callback=callback_graph,
        quantum_instance=QuantumInstance(
        Aer.get_backend("qasm_simulator"),
        shots=1000,
        seed_simulator=random_seed,
        seed_transpiler=random_seed,
        ),
    )
    
    cur_iter = 0
    train_X, train_y = get_sampled_train_data(
    batch_size=200)

    for iter_blk in range(1):
            # for iter_blk in range(3):  # 1 is not enough if you are reusing the parameters
            # for iter_blk in range((self.vqc_iterations-cur_iter)//10):  # original experiment

        vqc.fit(train_X, train_y)
        y_pred = vqc.predict(valid_X)
        final_pred = np.delete(y_pred, 1, 1)

        f1_score_weighted_avg = f1_score(
            valid_y.T[0], y_pred.T[0], average="weighted"
        )
        cur_iter += 10

        vqc.optimizer = COBYLA(maxiter=10)
        vqc.warm_start = True

        param_values = vqc.initial_point

    return f1_score_weighted_avg, param_values



def do_spsa(no_of_iter = 500, delta = 0.001, lr=0.001):

    dqn = DQN(max_depth*num_qubits,n_actions)
    dqn.load_state_dict(torch.load("./final_model/model.pt"))

    theta_t = dqn.state_dict()

    for i in range(102,no_of_iter+1):

        logger.debug(f"--------Iteration No: {i}---------")
        print(f"Iteration No: {i}")

        pert_t = deepcopy(theta_t)
        grad_t = deepcopy(theta_t)
        theta_next = deepcopy(theta_t)
        for param in theta_t:
            a = np.random.choice([1,-1],theta_t[param].size())
            pert_t[param] = theta_t[param] + torch.tensor(delta * a)
            grad_t[param] = (1/delta) * a

        curr_f1,curr_state = evaluation(theta_t)
        stats_buffer['f1_scores'].append(curr_f1)
        stats_buffer['states'].append(curr_state.tolist())

        print("F(theta):", curr_f1)
        logger.debug(f"F(theta):{curr_f1}")
        next_f1,_ = evaluation(pert_t)
        diff = next_f1 - curr_f1
        print("F(theta_perturbed):",next_f1 )
        logger.debug(f"F(theta_perturbed):{next_f1}" )

        wandb.log({"Iter":i,"F1 score": curr_f1})

        for param in theta_next:
            theta_next[param] = theta_t[param]+ lr * diff* grad_t[param]
        
        logger.debug("\n")

        if time() - start_time > 234000:
            filepath = "./json_objs/stats.json"
            json_object = json.dumps(stats_buffer)
            with open(filepath, "w") as outfile:
                outfile.write(json_object)
            break

    filepath = "./json_objs/stats.json"
    json_object = json.dumps(stats_buffer)
    with open(filepath, "w") as outfile:
        outfile.write(json_object)

    return theta_next



def evaluation(parameters):

    dqn = DQN(num_qubits*max_depth,n_actions)
    dqn.load_state_dict(parameters)
    epsilon = 0.1
    depth = 0
    param_vec = [Parameter(f"theta_{i}") for i in range(1000)]
    param_values = []

    curr_f1_max = 0
    curr_f1_state = None

    qc = QuantumCircuit(num_qubits)
    t=0
    added_param_counter = 0
    state = np.zeros((num_qubits,max_depth))
    flattened_state = state.flatten()
    prev_action = {i:float("inf") for i in range(2 * num_qubits)}

    while True:
        print(prev_action)

        action = select_action(dqn,flattened_state,epsilon,depth,
                      len(param_values),qc,prev_action).item()
        
        flattened_state,f1, terminated,param_values,qc,added_param_counter,state,depth =step(action, prev_action, param_values,qc,
         added_param_counter,param_vec,
         state,depth)
        
        print(qc)
        print(f1)
        
        t+=1

        if f1 > curr_f1_max:
            curr_f1_max = f1 
            curr_f1_state = deepcopy(state)
        
        if terminated:
            return curr_f1_max,curr_f1_state


with open('./json_objs/stats.json') as user_file:
    stats_buffer = json.load(user_file)

print("-------------------Start of SPSA---------------")
logger.debug("-------------------Start of SPSA---------------")


final_theta = do_spsa()
print("------------------End of SPSA-------------------\n")
logger.debug("------------------End of SPSA-------------------\n")

final_f1,_ = evaluation(final_theta)
print("Final_F1:",final_f1)
logger.debug(f"Final F1:{final_f1}")


torch.save(final_theta,f"./final_model/model.pt")