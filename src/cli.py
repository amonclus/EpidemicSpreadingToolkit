from input.graph_generator import choose_graph_source
from simulation.bootstrap import BootstrapPercolation
from analysis.graph_statistics import print_graph_statistics
from analysis.parameter_sweep import run_full_parameter_sweep
from visualization.visualization import animate_cascade
import random


def main():
    print("Welcome to the network risk analysis tool!")
    print("Choose graph source or run full parameter sweep:")
    print("1 - Generate graph")
    print("2 - Load graph from file")
    print("3 - Run full parameter sweep on synthetic graphs")
    
    choice = input("Enter choice: ")
    if choice == "3":
        print("\nRunning full parameter sweep (this may take some time)...")
        results = run_full_parameter_sweep()

        for graph_type, data in results.items():
            print(f"\nResults for {graph_type} graphs:")
            for entry in data:
                print(entry)
        return

    g = choose_graph_source(choice)
    if g is None:
        print(f"Failed to load graph. Exiting.")
        return

    print(f"Graph loaded with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")

    print("\nChoose analysis option:")
    print("1 - Run bootstrap cascade analysis")
    print("2 - Show graph structural statistics")

    option = input("Enter option: ")

    if option == "1":
        threshold = int(input("Enter bootstrap threshold (default 2): ") or "2")

        n = g.number_of_nodes()
        seed_fraction = float(
            input(f"Enter initial infection probability (fraction of nodes, 0.0-1.0, default 0.05): ") or "0.05"
        )
        seed_fraction = max(0.0, min(1.0, seed_fraction))
        seed_size = max(1, int(seed_fraction * n))
        print(f"Seed fraction: {seed_fraction} → {seed_size} initially infected node(s) out of {n}")

        sim = BootstrapPercolation(g, threshold)

        visualize = input("Do you want to visualize the cascade? (y/n, default n): ") or "n"
        record_sequence = visualize.lower() == "y"

        print("Running bootstrap percolation simulation...")
        seed_nodes = set(random.sample(list(g.nodes()), seed_size))

        # Run simulation for visualization with the seed nodes
        result, activation_sequence = sim.run(seed_nodes, record_sequence=record_sequence)

        # Show single-run result at the user's chosen seed fraction
        print(f"\nSingle-run result at seed fraction {seed_fraction}:")
        print(f"  Cascade fraction: {result.cascade_fraction:.4f}")
        print(f"  Rounds:           {result.time_to_cascade}")
        print(f"  Full cascade:     {result.is_full_cascade}")

        # Collect structural metrics (evaluated at the critical seed size)
        metrics = sim.collect_metrics(seed_size, num_trials=50, seed=42)

        print("\nStructural metrics (averaged over 50 trials):")
        if metrics.critical_seed_size == n:
            print("  Network is too sparse for cascades with the chosen threshold.")
            print("  No bootstrap cascade can propagate in this graph.")
            print(f"  Critical seed size:           {metrics.critical_seed_size} (all nodes)")
            print("  Cascade probability:          0.0000")
            print("  Time to cascade (avg rounds): 0.00")
            print("  Percolation threshold:        1.0000")
        else:
            print(f"  Cascade size (avg fraction):  {metrics.cascade_size:.4f}")
            print(f"  Critical seed size:           {metrics.critical_seed_size}")
            print(f"  Cascade probability:          {metrics.cascade_probability:.4f}")
            print(f"  Time to cascade (avg rounds): {metrics.time_to_cascade:.2f}")
            print(f"  Percolation threshold:        {metrics.percolation_threshold:.4f}")

            # Compute a final robustness score (composite metric)
            # Higher cascade probability and cascade size → lower robustness
            robustness = (1 - result.cascade_fraction) * (1 / (1 + result.time_to_cascade))
            print(f"\nFinal robustness score (0=fragile, 1=robust): {robustness:.4f}")

        # Animate if requested
        if record_sequence and activation_sequence:
            animate_cascade(g, activation_sequence)

    elif option == "2":
        print("\nComputing graph statistics...\n")
        print_graph_statistics(g)

    else:
        print("Invalid option.")


if __name__ == "__main__":
    main()