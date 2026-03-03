import numpy as np
import pennylane as qml
import copy
from abc import ABC, abstractmethod

# --- GA PARAMETERS ---
MAX_DEPTH       = 40
POP_SIZE        = 2000
NUM_GENERATIONS = 100
DEF_MUT_RATE    = 0.3
DEF_MUT_BOOST_COOLDOWN = 10
STAG_THRESH     = 15

TOURNAMENT = "tournament"
ROULETTE   = "roulette"

rng = np.random.default_rng()

# --- ALLOWED GATES ---
single_parametrised_gates = [] # [qml.RX, qml.RY, qml.RZ]
single_gates = [
        qml.Hadamard, 
        qml.PauliX, qml.PauliY, qml.PauliZ, 
        qml.S, qml.T,
        # single_parametrised_gates
    ]
multi_gates = [qml.CNOT, qml.CZ, qml.Toffoli, qml.MultiControlledX]

# ==========================================
# TASK INTERFACE
# ==========================================
class SynthesisTask(ABC):
    """
    Abstract base class for any quantum synthesis problem.
    The GA will only interact with this interface.
    """
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits # The task now owns the qubit count
        
    @abstractmethod
    def evaluate(self, structure) -> float: 
        pass

    @abstractmethod
    def print_result(self, best_structure): 
        pass

# ==========================================
# UNITARY DECOMPOSITION
# ==========================================
class UnitaryDecompositionTask(SynthesisTask):
    """
    Specific implementation for synthesizing a circuit that matches a Target Unitary Matrix.
    """
    def __init__(self, target_matrix, n_qubits):
        self.target = target_matrix
        self.n_qubits = n_qubits
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # We define the exec_circuit QNode inside the task so it binds to the task's device
        @qml.qnode(self.dev)
        def _exec_circuit(structure):
            for gate, wires, theta in structure:
                if gate in single_parametrised_gates:
                    gate(theta, wires=wires)
                else:
                    gate(wires=wires)
            return qml.state()
            
        self.exec_circuit = _exec_circuit

    def evaluate(self, structure) -> float:
        # Get matrix and calculate overlap
        U = np.array(qml.matrix(self.exec_circuit)(structure)) # type: ignore
        V = self.target
        d = U.shape[0]
        overlap = np.trace(V.conj().T @ U)
        
        # Calculate raw fidelity
        fidelity = np.abs(overlap / d)**2
        return fidelity

    def remove_global_phase(self, U):
        overlap = np.trace(self.target.conj().T @ U)
        phase = overlap / np.abs(overlap)   # unit complex number
        return U / phase

    def print_result(self, best_structure):
        U = qml.matrix(self.exec_circuit)(best_structure) # type: ignore
        U_fixed = self.remove_global_phase(U)

        print("Best matrix :")
        print(np.round(U_fixed, 2))
        print("\nTarget matrix :")
        print(np.round(self.target, 2))    
        print("\nSynthesized Circuit:")
        print(qml.draw(self.exec_circuit)(best_structure))


# ==========================================
# GENETIC ALGORITHM
# ==========================================
def gen_circuit_structure(n_qubits, single_gate_pref=0.5):
    depth = rng.choice(range(1, MAX_DEPTH + 1))
    if n_qubits < 2: single_gate_pref = 1
    structure = []

    for d in range(depth):
        theta = rng.uniform(0, 2 * np.pi)
        if rng.random() < single_gate_pref:
            wire = int(rng.choice(n_qubits))
            gate = rng.choice(single_gates)
            structure.append([gate, wire, theta]) 
        else:
            gate = rng.choice(multi_gates)
            
            # Use getattr to get num_wires, default to 2
            raw_needed = getattr(gate, "num_wires", 2)

            if not isinstance(raw_needed, int):
                needed_wires = rng.choice(range(2, n_qubits + 1))
            else:
                needed_wires = min(raw_needed, n_qubits)
            
            wires = rng.choice(n_qubits, size=needed_wires, replace=False).tolist()
            structure.append([gate, wires, theta])
            
    return structure

def create_population(pop_size, task):
    population = []
    fitnesses = []
    for i in range(pop_size):
        structure = gen_circuit_structure(task.n_qubits)
        population.append(structure)
        fitness = task.evaluate(structure)
        fitnesses.append(fitness)
    return population, fitnesses

def sort_pop(population, fitnesses):
    idx = np.argsort(fitnesses)[::-1]
    fitnesses = np.array(fitnesses)[idx]
    population = np.array(population, dtype=object)[idx]
    return population, fitnesses

def select_parents(population, fitnesses, selection_type=TOURNAMENT):
    if selection_type == ROULETTE:
        probs = np.array(fitnesses)
        probs = probs / probs.sum()
        parents_idx = np.random.choice(len(fitnesses), size=2, p=probs, replace=False)
        return population[parents_idx[0]], population[parents_idx[1]]
    
    if selection_type == TOURNAMENT :
        tournament_size = 5
        def pick_one():
            candidates = rng.choice(len(fitnesses), size=tournament_size)
            best_idx = candidates[np.argmax(fitnesses[candidates])]
            return population[best_idx]
        return pick_one(), pick_one()

def crossover(parent1, parent2):
    m, n = len(parent1), len(parent2)
    max_len = max(2, (m + n) // 2)    
    len1 = rng.integers(1, max_len + 1)
    len2 = rng.integers(1, max_len + 1)
    child1 = parent1[:len1] + parent2[-len2:]
    child2 = parent2[:len1] + parent1[-len2:]
    return copy.deepcopy(child1), copy.deepcopy(child2)

def mutate(ind, n_qubits, mutation_rate=DEF_MUT_RATE):
    ind = copy.deepcopy(ind)
    if rng.random() > mutation_rate: 
        return ind

    # Angle mutation
    for gene in ind:
        if len(gene) == 3 and gene[0] in single_parametrised_gates:
            if rng.random() < 0.6:
                gene[2] = (gene[2] + rng.normal(0, 0.3)) % (2 * np.pi)
    
    if not ind: 
        return ind
    idx = rng.integers(0, len(ind))
    
    # Gate/Wire Mutation
    if rng.random() < 0.3:
        is_single = not isinstance(ind[idx][1], list)
        new_gate = rng.choice(single_gates) if is_single else rng.choice(multi_gates)
        ind[idx][0] = new_gate
        
        # Robustly determine wires for the new gate
        raw_needed = getattr(new_gate, "num_wires", 1)
        if not isinstance(raw_needed, int):
            needed = rng.choice(range(2, n_qubits + 1))
        else:
            needed = raw_needed
            
        if needed == 1:
            ind[idx][1] = int(rng.choice(n_qubits))
        else:
            n_to_pick = min(needed, n_qubits)
            ind[idx][1] = rng.choice(n_qubits, size=n_to_pick, replace=False).tolist()

    # Insert/Delete Mutation
    if rng.random() < 0.1:
        if rng.random() < 0.5 and len(ind) > 1:
            del ind[idx]
        else:
            ind.append(gen_circuit_structure(n_qubits, single_gate_pref=0.5)[0])
            
    return ind

def eval_pop(population, task):
    fitnesses = []
    for ind in population :
        # Abstract evaluation
        fitnesses.append(task.evaluate(ind)) 
    return fitnesses

def genetic_algorithm(task: SynthesisTask, elitism_frac=0.2):
    N_QUBITS = task.n_qubits
    population, fitnesses = create_population(pop_size=POP_SIZE, task=task)
    population, fitnesses = sort_pop(population, fitnesses)

    best_fitnesses = [0]
    mut_rate = DEF_MUT_RATE
    stag_count = 0
    mut_boost_cooldown = DEF_MUT_BOOST_COOLDOWN
    generation = 0

    while (generation < NUM_GENERATIONS) :
        if generation % 10 == 0 :
            print(f"Generation   {generation} | Best fitness: {fitnesses[0]:.6f}")
        
        if fitnesses[0] == best_fitnesses[-1]: 
            stag_count += 1
        else: 
            stag_count = 0
            if (mut_rate > DEF_MUT_RATE) and (mut_boost_cooldown == 0):
                mut_rate *= 0.9
                print("     Reducing mutation rate")
                mut_boost_cooldown = DEF_MUT_BOOST_COOLDOWN
                
        mut_boost_cooldown = max(mut_boost_cooldown-1, 0)

        if (stag_count > STAG_THRESH) and (mut_boost_cooldown == 0):
            mut_rate = min(1.2*mut_rate, 0.8)
            print("     Boosting mutation rate")
            mut_boost_cooldown = DEF_MUT_BOOST_COOLDOWN

        best_fitnesses.append(fitnesses[0])
        num_elites = int(elitism_frac * POP_SIZE)
        children = [copy.deepcopy(ind) for ind in population[:num_elites]]

        for _ in range(int((POP_SIZE - num_elites) / 2)):
            parent1, parent2 = select_parents(population, fitnesses, TOURNAMENT) # type: ignore
            child1, child2 = crossover(parent1, parent2)
            children.append(mutate(child1, N_QUBITS))
            children.append(mutate(child2, N_QUBITS))

        population = children
        fitnesses = eval_pop(population, task) # Pass task here
        population, fitnesses = sort_pop(population, fitnesses)
        generation += 1

    population, fitnesses = sort_pop(population, fitnesses)

    print(f"\n{generation} generations completed")
    print(f"Final Best fitness value: {fitnesses[0]:.6f}")
    
    task.print_result(population[0]) 


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    def random_unitary(d):
        Z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2)
        Q, R = np.linalg.qr(Z)
        phases = np.diag(R) / np.abs(np.diag(R))
        return Q * phases
    
    N_QUBITS = 2
    print("--- Starting Unitary Decomposition Task ---")
    my_target = random_unitary(2**N_QUBITS)
    
    # Instantiate the specific task
    my_task = UnitaryDecompositionTask(target_matrix=my_target, n_qubits=N_QUBITS)
    # Run the synthesizer
    genetic_algorithm(task=my_task)