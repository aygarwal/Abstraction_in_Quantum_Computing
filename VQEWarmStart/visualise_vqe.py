import json
import matplotlib.pyplot as plt
import numpy as np

with open("vqe_results.json", "r") as f:
    data = json.load(f)

# Example 1: Convergence Plot for a specific run
plt.figure(figsize=(10, 6))
for run in data:
    if run["n_qubits"] == 5: # Filter for 5 qubits
        plt.plot(run["history"], label=f"Layers: {run['n_layers']}")

plt.axhline(y=data[0]["true_energy"], color='r', linestyle='--', label="True Ground Energy")
plt.title("VQE Convergence: 5 Qubits")
plt.xlabel("Optimization Steps")
plt.ylabel("Energy")
plt.legend()
plt.grid(True)
plt.show()

# Example 2: Success Rate vs Qubits
qubits = [run["n_qubits"] for run in data if run["n_layers"] == 1]
fitness = [run["fitness"] * 100 for run in data if run["n_layers"] == 1]

plt.figure(figsize=(8, 5))
plt.bar(qubits, fitness, color='skyblue')
plt.ylabel("Accuracy (%)")
plt.xlabel("Number of Qubits")
plt.title("VQE Accuracy with 1 Layer")
plt.ylim(0, 110)
plt.show()