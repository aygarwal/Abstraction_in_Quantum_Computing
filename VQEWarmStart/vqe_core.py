import pennylane as qml #type: ignore
from genetic_algorithm import SynthesisTask, genetic_algorithm
import numpy as np
from pennylane import numpy as pnp

# --- ALLOWED GATES ---
single_parametrised_gates = [qml.RX, qml.RY, qml.RZ]
single_gates = [
        qml.Hadamard, 
        qml.PauliX, qml.PauliY, qml.PauliZ, 
        qml.S, qml.T,
        qml.RX, qml.RY, qml.RZ
    ]
multi_gates = [qml.CNOT, qml.CZ, qml.Toffoli, qml.MultiControlledX]

class VQETask(SynthesisTask):
    def __init__(self, hamiltonian):
        gates = [single_gates, single_parametrised_gates, multi_gates]
        n_qubits = max([max(o.wires) for o in hamiltonian.ops]) + 1
        super().__init__(n_qubits, gates)
        
        self.H = hamiltonian
        self.dev = qml.device('default.qubit', wires=self.n_qubits)

        @qml.qnode(self.dev)
        def _circuit(structure):
            for gate, wires, theta in structure:
                target_wires = wires if isinstance(wires, list) else int(wires)
                if gate in single_parametrised_gates:
                    gate(theta, wires=target_wires)
                else: gate(wires=target_wires)
            return qml.expval(self.H)
        self.qnode = _circuit

    def evaluate(self, structure):
        energy = float(self.qnode(structure))
        return -energy - (len(structure) * 0.005)

    def print_result(self, best_structure):
        print(f"VQE Result for {self.n_qubits} Qubits")
        print(qml.draw(self.qnode)(best_structure))

def create_Hamiltonian(n_qubits=4, connectivity_prob=0.6):
    coeffs = []
    obs = []

    # --- Random pairwise interactions ---
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):

            if np.random.rand() < connectivity_prob:

                Jx = np.random.uniform(-1, 1)
                Jy = np.random.uniform(-1, 1)
                Jz = np.random.uniform(-1, 1)

                coeffs.append(Jx)
                obs.append(qml.PauliX(i) @ qml.PauliX(j))

                coeffs.append(Jy)
                obs.append(qml.PauliY(i) @ qml.PauliY(j))

                coeffs.append(Jz)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

    # --- Random local fields ---
    for i in range(n_qubits):

        hx = np.random.uniform(-1, 1)
        hy = np.random.uniform(-1, 1)
        hz = np.random.uniform(-1, 1)

        coeffs.append(hx)
        obs.append(qml.PauliX(i))

        coeffs.append(hy)
        obs.append(qml.PauliY(i))

        coeffs.append(hz)
        obs.append(qml.PauliZ(i))

    return qml.Hamiltonian(coeffs, obs)

def ground_state_energy(H):
    H_mat = np.array(qml.matrix(H))
    eigenvalues = np.linalg.eigvalsh(H_mat)
    return np.min(eigenvalues)

def run_vqe (task, initial_ansatz, opt) :
    # Identify which gates in the GA result are parameterized
    trainable_indices = [i for i, gate in enumerate(initial_ansatz) 
                        if gate[0] in single_parametrised_gates]
    
    # Extract initial values from GA as starting parameters
    raw_params = [float(initial_ansatz[i][2]) for i in trainable_indices]
    params = pnp.array(raw_params, requires_grad=True)

    if len(params) == 0:
        print("This ansatz has no trainable parameters. Skipping VQE refinement.")
        return

    # --- B. Define a Parameter-Specific QNode ---
    @qml.qnode(task.dev)
    def vqe_circuit(current_params):
        p_idx = 0
        for i, (gate, wires, theta) in enumerate(initial_ansatz):
            if i in trainable_indices:
                # Use the optimizer's parameter instead of the GA's fixed theta
                gate(current_params[p_idx], wires=wires)
                p_idx += 1
            else:
                gate(wires=wires)
        return qml.expval(task.H)

    # --- C. Run Optimization ---
    max_iters = 100
    
    print(f"\nStarting refinement on GA-found ansatz (Initial Energy: {vqe_circuit(params):.4f})")

    def cost_fn (p) : return vqe_circuit(p)

    for i in range(max_iters):
        params, energy = opt.step_and_cost(cost_fn, params)
        
        if i % 10 == 0:
            print(f"  Step {i}: Energy = {energy:.8f}")