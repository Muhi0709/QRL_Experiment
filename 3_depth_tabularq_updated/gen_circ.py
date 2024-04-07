import numpy as np
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
from qiskit import qpy


circ_list = [QuantumCircuit(6),]

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

for i in range(10):
    circ = QuantumCircuit(6)
    if i<3:
        circ.rx(Parameter('theta_0'),d1[i])
    elif i<6:
        circ.ry(Parameter('theta_0'),d1[i])

    else:
        circ.cx(d1[i][0],d1[i][1])
    
    circ_list.append(circ)


for i in range(10):
    for j in range(10):
        circ = QuantumCircuit(6)
        if i<3:
            circ.rx(Parameter('theta_0'),d1[i])
        elif i<6:
            circ.ry(Parameter('theta_0'),d1[i])

        else:
            circ.cx(d1[i][0],d1[i][1])

            
        if j<3:
            circ.ry(Parameter('theta_1'),d2[j])

        elif j<6:
            circ.rz(Parameter('theta_1'),d2[j])

        else:
            circ.cx(d2[j][0],d2[j][1])
        
        circ_list.append(circ)


for i in range(10):
    for j in range(10):
        for k in range(10):
            print(i,j,k)
            
            state = np.zeros((6,3),dtype=np.int32)
            circ = QuantumCircuit(6)
            if i<3:
                state[d1[i]][0] = 1
                circ.rx(Parameter('theta_0'),d1[i])
            elif i<6:
                state[d1[i]][0] = 2
                circ.ry(Parameter('theta_0'),d1[i])

            else:
                state[d1[i][0]][0]=4
                state[d1[i][1]][0]=5
                circ.cx(d1[i][0],d1[i][1])

            
            if j<3:
                state[d2[j]][1] = 2
                circ.ry(Parameter('theta_1'),d2[j])

            elif j<6:
                state[d2[j]][1] = 3
                circ.rz(Parameter('theta_1'),d2[j])

            else:
                state[d2[j][0]][1] = 4
                state[d2[j][1]][1] = 5
                circ.cx(d2[j][0],d2[j][1])
            
            if k<3:
                state[d3[k]][2] = 3
                circ.rz(Parameter('theta_2'),d3[k])

            elif k<6:
                state[d3[k]][2] = 1
                circ.rx(Parameter('theta_2'),d3[k])

            else:
                circ.cx(d3[k][0],d3[k][1])
                state[d3[k][0]][2] = 4
                state[d3[k][1]][2] = 5
            
            circ_list.append(circ)

with open('3depth_circ_list.qpy', 'wb') as fd:
    qpy.dump(circ_list, fd)

