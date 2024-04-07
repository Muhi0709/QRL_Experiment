import numpy as np
import time

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

from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

def create_one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]


dim = 64
data_type = "NvsBU"

df_train = pd.read_csv(
"/content/drive/MyDrive/3depth_value_iter/Data/gnnfeatures_dim_64/features_bracs_hact_2_classes_pna_NvsBU_train.csv"
).to_numpy()

df_valid = pd.read_csv(
"/content/drive/MyDrive/3depth_value_iter/Data/gnnfeatures_dim_64/features_bracs_hact_2_classes_pna_NvsBU_val.csv"
).to_numpy()

df_test = pd.read_csv(
"/content/drive/MyDrive/3depth_value_iter/Data/gnnfeatures_dim_64/features_bracs_hact_2_classes_pna_NvsBU_test.csv"
).to_numpy()

valid_X = df_valid[:, 3:]
valid_y = df_valid[:, 2].astype(int)
test_X = df_test[:, 3:]
test_y = df_test[:, 2].astype(int)

valid_y = create_one_hot(valid_y, 2)
test_y = create_one_hot(test_y, 2)


def get_sampled_train_data():  #entire dataset,just shuffled
    np.random.shuffle(df_train)

    train_X = df_train[:, 3:]
    train_y = df_train[:, 2].astype(int)
    train_y = create_one_hot(train_y, 2)

    return train_X, train_y

objective_values=[]
def callback_graph(_, objective_value):
    # clear_output(wait=True)
    objective_values.append(objective_value)

def perform_vqc(ansatz,random_seed = 123):
    
    vqc = VQC(
        feature_map= RawFeatureVector(dim),
        ansatz= ansatz,
        optimizer=COBYLA(maxiter=10),
        callback=callback_graph,
        quantum_instance=QuantumInstance(
                    Aer.get_backend("qasm_simulator"),
                    shots=1028,
                    seed_simulator=random_seed,
                    seed_transpiler=random_seed,
                ),
            )
    cur_iter = 0
    
    train_X, train_y = get_sampled_train_data()

    for iter_blk in range(1):

        vqc.fit(train_X, train_y)
        y_pred = vqc.predict(valid_X)
        final_pred = np.delete(y_pred, 1, 1)

        f1_score_weighted_avg = f1_score(
            valid_y.T[0], y_pred.T[0], average="weighted"
        )
        cur_iter += 10

        vqc.optimizer = COBYLA(maxiter=10)
        vqc.warm_start = True


    return f1_score_weighted_avg




def value_iteration(max_iter=1000,value_table=np.zeros(1),gamma = 0.99,circ_list=[],time_limit = 5000,random_seed=12345678,start=1110):
    num_states = 10**3 + 10**2 + 10 + 1
    start_time = time.time()
    i=0
    for s in range(start,-1,-1):   
        print(i,s)        #going in reverse is faster
        if s<111:
            maxi = -float("inf")
            for a in range(1,11):
                next_s = 10*s+ a
                rew = perform_vqc(circ_list[next_s],random_seed)
                if (rew+gamma*value_table[next_s]) > maxi:
                    maxi = (rew+gamma*value_table[next_s])
            value_table[s] = maxi
        else:
            rew = perform_vqc(circ_list[s],random_seed)
            value_table[s] = (rew)/(1-gamma)
        
        np.save("/content/drive/MyDrive/3depth_value_iter/3depth_value.npy",value_table)
        if time.time()- start_time> time_limit:
            break
    
                

