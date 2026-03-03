import pennylane as qml
from TaskAgnosticGA import SynthesisTask, genetic_algorithm, single_parametrised_gates

class VQETask(SynthesisTask):
    def __init__(self, hamiltonian):
        # Find max wire index in Hamiltonian to determine n_qubits
        n_qubits = max([max(o.wires) for o in hamiltonian.ops]) + 1
        super().__init__(n_qubits)
        
        self.H = hamiltonian
        self.dev = qml.device('default.qubit', wires=self.n_qubits)

        @qml.qnode(self.dev)
        def _circuit(structure):
            for gate, wires, theta in structure:
                if gate in single_parametrised_gates: gate(theta, wires=wires)
                else: gate(wires=wires)
            return qml.expval(self.H)
        self.qnode = _circuit

    def evaluate(self, structure):
        energy = float(self.qnode(structure))
        return -energy - (len(structure) * 0.005)

    def print_result(self, best_structure):
        print(f"VQE Result for {self.n_qubits} Qubits")
        print(qml.draw(self.qnode)(best_structure))

def create_Hamiltonian (n_qubits=3) :
    # Parameters for a 3-qubit TFIM
    coeffs = []
    obs = []
    J = 1.0  # Coupling constant

    for i in range(n_qubits - 1):
        # X-X coupling
        coeffs.append(J)
        obs.append(qml.PauliX(i) @ qml.PauliX(i+1))
        
        # Y-Y coupling
        coeffs.append(J)
        obs.append(qml.PauliY(i) @ qml.PauliY(i+1))
        
        # Z-Z coupling
        coeffs.append(J)
        obs.append(qml.PauliZ(i) @ qml.PauliZ(i+1))

    H = qml.Hamiltonian(coeffs, obs)
    return H

if __name__ == "__main__":
    # Create a 3-qubit Hamiltonian
    H = create_Hamiltonian(4)
    print(H)
    task = VQETask(H)
    genetic_algorithm(task)