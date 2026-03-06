"""
1. Create random Hamiltonian / Use known common Hamiltonian
2. Use GA to find likely good initial ansatz(s)
3. Run VQE loop now with warm start 
"""

from vqe_core import create_Hamiltonian, ground_state_energy, VQETask, run_vqe
from genetic_algorithm import genetic_algorithm
import pennylane as qml #type: ignore


# 1. Create question
print("\n ----------- HAMILTONIAN -----------")
H = create_Hamiltonian(n_qubits=7)
print(H)
print("True ground energy = (-)", ground_state_energy(H))
task = VQETask(H)

# 2. Run GA to find likely good initial ansatz
print("\n -------- GENETIC ALGORITHM --------")
warm_start_results = genetic_algorithm(
        task, 
        n_results=10,
        max_depth=30,
        pop_size=1000,
        num_generations=50
    )

# 3. VQE loop
print("\n --------------- VQE ---------------")
for warm_init in warm_start_results :
    opt = qml.AdamOptimizer(stepsize=0.1)
    run_vqe(task, warm_init, opt)