import pennylane as qml #type: ignore
from pennylane import numpy as pnp
from vqe_core import create_Hamiltonian, ground_state_energy

# 1. Create Hamiltonian
n_qubits = 12
H = create_Hamiltonian(n_qubits=n_qubits)
dev = qml.device("default.qubit", wires=n_qubits)

# 2. Define standard templates as QNodes
# We wrap them so they all take 'params' as input
@qml.qnode(dev)
def strongly_entangling_ansatz(params):
    qml.StronglyEntanglingLayers(weights=params, wires=range(n_qubits))
    return qml.expval(H)

@qml.qnode(dev)
def basic_entangler_ansatz(params):
    qml.BasicEntanglerLayers(weights=params, wires=range(n_qubits))
    return qml.expval(H)

# 3. Setup Initial Parameters
n_layers = 75

# Templates have helper functions to tell you the required parameter shape
ansatz_list = [
    {
        "name": "Strongly Entangling",
        "qnode": strongly_entangling_ansatz,
        "params": pnp.random.random(qml.StronglyEntanglingLayers.shape(n_layers, n_qubits), requires_grad=True) # type: ignore
    },
    {
        "name": "Basic Entangler",
        "qnode": basic_entangler_ansatz,
        "params": pnp.random.random(qml.BasicEntanglerLayers.shape(n_layers, n_qubits), requires_grad=True) # type: ignore
    }
]

# 4. Run VQE Loop
print(f"True ground energy: {ground_state_energy(H):.6f}")

for entry in ansatz_list:
    print(f"\n--- Running {entry['name']} ---")
    params = entry['params']
    opt = qml.AdamOptimizer(stepsize=0.2)
    
    for i in range(500):
        params, energy = opt.step_and_cost(entry['qnode'], params)
        if i % 20 == 0:
            print(f"Step {i}: Energy = {energy:.6f}")