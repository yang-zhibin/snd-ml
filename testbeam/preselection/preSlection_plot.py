import argparse
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Test script with plot generation.")
    parser.add_argument("-b", "--beam_type", required=True, help="Type of beam (e.g., proton, kaon)")
    parser.add_argument("-e", "--energy", required=True, type=float, help="Beam energy in GeV")

    args = parser.parse_args()

    beam = args.beam_type
    energy = args.energy

    print(f"Beam type: {beam}")
    print(f"Energy: {energy} GeV")

    # Simulate some dummy detector layer data
    layers = np.arange(1, 11)  # 10 layers
    # Fake energy deposition: depends on input energy + some randomness
    energy_deposit = np.random.normal(loc=energy / 10, scale=0.5, size=len(layers))

    # Plot
    plt.figure()
    plt.bar(layers, energy_deposit)
    plt.xlabel("Detector Layer")
    plt.ylabel("Energy Deposited (a.u.)")
    plt.title(f"{beam} beam at {energy} GeV")
    plt.tight_layout()

    # Save plot
    plot_filename = f"./plots/test_plot_{beam}_{energy}.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")

if __name__ == "__main__":
    main()