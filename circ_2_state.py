from qiskit import QuantumCircuit

import numpy as np
def circ_to_state(circ): #circ-QuantumCircuit Object-6 qubits
    num_qubits = 6
    max_depth = 20
    state = np.zeros((num_qubits,max_depth))

    counter=0
    for inst in circ.data:
        if inst[0].name == "cx":
            control = inst[1][0].index
            target = inst[1][1].index
            state[control][counter] = 4
            state[target][counter] = 5

        elif inst[0].name == "rx":
            param = inst[0].params[0]  #parameter can also be accessed
            qubit = inst[1][0].index
            state[qubit][counter] = 1
                    
        elif inst[0].name == "rz":
            param = inst[0].params[0]
            qubit = inst[1][0].index
            state[qubit][counter] = 3

        elif inst[0].name == "ry":
            param = inst[0].params[0]
            qubit = inst[1][0].index
            state[qubit][counter] = 2

        counter += 1
    return state


circ = QuantumCircuit(6)
circ.cx(0,1)
circ.rx(np.pi,0)
circ.rz(np.pi,2)
circ.ry(np.pi,3)

print(circ_to_state(circ))