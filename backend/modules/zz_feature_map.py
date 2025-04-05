from qiskit.circuit.library import ZZFeatureMap
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt

def generate_feature_map_image(data, filename="circuit.svg", reps=2):
    """
    Generates and saves a feature map circuit image.

    Parameters:
    - data (list): Input feature vector.
    - filename (str): Output file name (default: "circuit.svg").
    - reps (int): Number of repetitions in ZZFeatureMap (default: 2).
    """
    num_qubits = len(data)  # Number of qubits = length of data
    feature_map = ZZFeatureMap(num_qubits, reps=reps).assign_parameters(data)

    # Draw and save the circuit
    fig, ax = plt.subplots()
    circuit_drawer(feature_map.decompose(), output='mpl', ax=ax)
    plt.savefig(filename, format="svg")
    plt.close(fig)  # Close the figure to prevent display issues
    print(f"Feature map saved as {filename}")
    return filename

if __name__ == '__main__':
  # Example usage
  generate_feature_map_image([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])
