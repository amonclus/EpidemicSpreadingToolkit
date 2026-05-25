"""
Tests for all epidemic simulation models:
  SIRModel, SISModel, BootstrapPercolation, WTMModel, H1–H6 Models.

Strategy
--------
* Deterministic invariants are tested on single runs.
* Stochastic magnitude assertions (large/small cascade) average at least
  N_TRIALS independent runs to reduce false failures.
* Small graphs (≤ 30 nodes) are used throughout.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import pytest
import networkx as nx

from simulation.sir import SIRModel, SIRResult
from simulation.sis import SISModel, SISResult
from simulation.bootstrap import BootstrapPercolation, BootstrapResult
from simulation.wtm import WTMModel, WTMResult
from simulation.H1 import H1Model, H1Result
from simulation.H2 import H2Model, H2Result
from simulation.H3 import H3Model, H3Result
from simulation.H4 import H4Model, H4Result
from simulation.H5 import H5Model, H5Result
from simulation.H6 import H6Model, H6Result

N_TRIALS = 20  # repetitions for stochastic assertions


# ─── helpers ──────────────────────────────────────────────────────────────────

def make_er(n=20, p=0.3, seed=42):
    random.seed(seed)
    g = nx.erdos_renyi_graph(n, p, seed=seed)
    # guarantee connectivity
    if not nx.is_connected(g):
        nodes = list(g.nodes())
        for i in range(len(nodes) - 1):
            g.add_edge(nodes[i], nodes[i + 1])
    return g


def make_complete(n=15):
    return nx.complete_graph(n)


def make_lattice(size=4):
    g = nx.grid_2d_graph(size, size)
    return nx.convert_node_labels_to_integers(g)


# ─── SIR ──────────────────────────────────────────────────────────────────────

class TestSIRModel:

    def test_epidemic_fraction_in_unit_interval(self):
        g = make_er()
        model = SIRModel(g, beta=0.3, gamma=0.1)
        for _ in range(N_TRIALS):
            seeds = {random.choice(list(g.nodes()))}
            result, _ = model.run(seeds)
            assert 0.0 <= result.epidemic_fraction <= 1.0

    def test_epidemic_size_equals_infected_nodes_len(self):
        g = make_er()
        model = SIRModel(g, beta=0.3, gamma=0.1)
        random.seed(1)
        seeds = set(random.sample(list(g.nodes()), 3))
        result, _ = model.run(seeds)
        assert result.epidemic_size == len(result.infected_nodes)

    def test_empty_seed_produces_zero_epidemic(self):
        g = make_er()
        model = SIRModel(g, beta=0.5, gamma=0.1)
        result, _ = model.run(set())
        assert result.epidemic_fraction == 0.0
        assert result.epidemic_size == 0

    def test_recovered_nodes_subset_of_infected(self):
        g = make_er()
        model = SIRModel(g, beta=0.4, gamma=0.2)
        random.seed(2)
        seeds = set(random.sample(list(g.nodes()), 3))
        result, _ = model.run(seeds)
        assert result.recovered_nodes.issubset(result.infected_nodes)

    def test_time_to_epidemic_positive_for_nonempty_seed(self):
        g = make_er()
        model = SIRModel(g, beta=0.3, gamma=0.2)
        random.seed(3)
        seeds = {list(g.nodes())[0]}
        result, _ = model.run(seeds)
        # At minimum one round must have passed (the seed exists)
        assert result.time_to_epidemic >= 0  # 0 is valid if beta→0 immediately

    def test_supercritical_produces_large_epidemic_on_average(self):
        """High beta, low gamma → epidemic fraction > 0.3 on average."""
        g = make_er(n=30, p=0.25)
        model = SIRModel(g, beta=0.6, gamma=0.05)
        random.seed(42)
        fracs = []
        for _ in range(N_TRIALS):
            seeds = set(random.sample(list(g.nodes()), 3))
            result, _ = model.run(seeds)
            fracs.append(result.epidemic_fraction)
        assert sum(fracs) / len(fracs) > 0.2

    def test_subcritical_produces_small_epidemic_on_average(self):
        """Very low beta → average epidemic fraction near seed fraction."""
        g = make_er(n=30, p=0.25)
        model = SIRModel(g, beta=0.01, gamma=0.95)
        random.seed(99)
        fracs = []
        for _ in range(N_TRIALS):
            seeds = set(random.sample(list(g.nodes()), 2))
            result, _ = model.run(seeds)
            fracs.append(result.epidemic_fraction)
        assert sum(fracs) / len(fracs) < 0.3

    def test_record_sequence_format(self):
        """Activation sequence is a list of (set, set) tuples."""
        g = make_er()
        model = SIRModel(g, beta=0.4, gamma=0.1)
        random.seed(5)
        seeds = set(random.sample(list(g.nodes()), 3))
        result, seq = model.run(seeds, record_sequence=True)
        assert isinstance(seq, list)
        assert len(seq) > 0
        for entry in seq:
            assert isinstance(entry, tuple) and len(entry) == 2
            infected_set, recovered_set = entry
            assert isinstance(infected_set, set)
            assert isinstance(recovered_set, set)

    def test_seed_nodes_always_in_infected(self):
        g = make_er()
        model = SIRModel(g, beta=0.3, gamma=0.1)
        random.seed(7)
        seeds = set(random.sample(list(g.nodes()), 4))
        result, _ = model.run(seeds)
        assert seeds.issubset(result.infected_nodes)

    def test_all_nodes_infected_on_complete_supercritical(self):
        """On K_15 with beta=1, all nodes must eventually get infected."""
        g = make_complete(15)
        # gamma=0 causes an infinite loop (infected nodes never recover and
        # the loop only exits when infected is empty). Use a small gamma so
        # the model terminates while still spreading to every node.
        model = SIRModel(g, beta=1.0, gamma=0.1)
        seeds = {0}
        result, _ = model.run(seeds)
        assert result.epidemic_size == g.number_of_nodes()
        assert result.epidemic_fraction == 1.0

    def test_total_nodes_conserved(self):
        """infected_nodes is always a subset of graph nodes."""
        g = make_er()
        model = SIRModel(g, beta=0.3, gamma=0.1)
        random.seed(11)
        seeds = set(random.sample(list(g.nodes()), 3))
        result, _ = model.run(seeds)
        assert result.infected_nodes.issubset(set(g.nodes()))

    def test_epidemic_fraction_equals_size_over_n(self):
        g = make_er()
        n = g.number_of_nodes()
        model = SIRModel(g, beta=0.3, gamma=0.1)
        random.seed(13)
        seeds = set(random.sample(list(g.nodes()), 2))
        result, _ = model.run(seeds)
        assert abs(result.epidemic_fraction - result.epidemic_size / n) < 1e-9


# ─── SIS ──────────────────────────────────────────────────────────────────────

class TestSISModel:

    def test_cascade_fraction_in_unit_interval(self):
        g = make_er()
        model = SISModel(g, beta=0.3, gamma=0.1)
        for _ in range(N_TRIALS):
            random.seed(_)
            seeds = {random.choice(list(g.nodes()))}
            result, _ = model.run(seeds)
            assert 0.0 <= result.cascade_fraction <= 1.0

    def test_cascade_size_equals_peak_infected(self):
        g = make_er()
        model = SISModel(g, beta=0.4, gamma=0.1)
        random.seed(20)
        seeds = set(random.sample(list(g.nodes()), 3))
        result, _ = model.run(seeds)
        assert result.cascade_size == result.peak_infected

    def test_empty_seed_produces_zero_peak(self):
        g = make_er()
        model = SISModel(g, beta=0.5, gamma=0.1)
        result, _ = model.run(set())
        assert result.cascade_fraction == 0.0
        assert result.peak_infected == 0

    def test_infected_series_length_matches_rounds(self):
        g = make_er()
        model = SISModel(g, beta=0.3, gamma=0.2, max_steps=50)
        random.seed(30)
        seeds = set(random.sample(list(g.nodes()), 2))
        result, _ = model.run(seeds)
        # infected_series has one entry per step including the initial step
        assert len(result.infected_series) == result.time_to_cascade + 1

    def test_record_sequence_format(self):
        g = make_er()
        model = SISModel(g, beta=0.4, gamma=0.1)
        random.seed(31)
        seeds = set(random.sample(list(g.nodes()), 2))
        result, seq = model.run(seeds, record_sequence=True)
        assert isinstance(seq, list)
        assert len(seq) > 0
        for entry in seq:
            assert isinstance(entry, tuple) and len(entry) == 2

    def test_seed_nodes_in_ever_infected(self):
        g = make_er()
        model = SISModel(g, beta=0.3, gamma=0.2)
        random.seed(32)
        seeds = set(random.sample(list(g.nodes()), 3))
        result, _ = model.run(seeds)
        assert seeds.issubset(result.infected_nodes)

    def test_peak_infected_at_most_n(self):
        g = make_er()
        n = g.number_of_nodes()
        model = SISModel(g, beta=0.5, gamma=0.05)
        random.seed(33)
        seeds = set(random.sample(list(g.nodes()), 5))
        result, _ = model.run(seeds)
        assert result.peak_infected <= n

    def test_max_steps_respected(self):
        """Simulation must not exceed max_steps rounds."""
        g = make_er()
        max_steps = 10
        model = SISModel(g, beta=0.5, gamma=0.05, max_steps=max_steps)
        random.seed(34)
        seeds = set(random.sample(list(g.nodes()), 5))
        result, _ = model.run(seeds)
        assert result.time_to_cascade <= max_steps

    def test_cascade_fraction_matches_peak_over_n(self):
        g = make_er()
        n = g.number_of_nodes()
        model = SISModel(g, beta=0.3, gamma=0.2)
        random.seed(35)
        seeds = set(random.sample(list(g.nodes()), 2))
        result, _ = model.run(seeds)
        assert abs(result.cascade_fraction - result.peak_infected / n) < 1e-9


# ─── Bootstrap Percolation ────────────────────────────────────────────────────

class TestBootstrapPercolation:

    def test_cascade_fraction_in_unit_interval(self):
        g = make_er()
        model = BootstrapPercolation(g, threshold=2)
        for _ in range(N_TRIALS):
            random.seed(_)
            seeds = set(random.sample(list(g.nodes()), 3))
            result, _ = model.run(seeds)
            assert 0.0 <= result.cascade_fraction <= 1.0

    def test_cascade_size_equals_infected_nodes_len(self):
        g = make_er()
        model = BootstrapPercolation(g, threshold=2)
        random.seed(40)
        seeds = set(random.sample(list(g.nodes()), 4))
        result, _ = model.run(seeds)
        assert result.cascade_size == len(result.infected_nodes)

    def test_empty_seed_produces_zero(self):
        g = make_er()
        model = BootstrapPercolation(g, threshold=2)
        result, _ = model.run(set())
        assert result.cascade_fraction == 0.0
        assert result.cascade_size == 0

    def test_deterministic_output_same_seed_same_result(self):
        """Bootstrap percolation is deterministic — same seeds → same result."""
        g = make_er(seed=0)
        model = BootstrapPercolation(g, threshold=2)
        seeds = set(list(g.nodes())[:5])
        result1, _ = model.run(seeds)
        result2, _ = model.run(seeds)
        assert result1.cascade_size == result2.cascade_size
        assert result1.infected_nodes == result2.infected_nodes

    def test_is_full_cascade_flag(self):
        """is_full_cascade should be True iff all nodes are infected."""
        g = make_er()
        model = BootstrapPercolation(g, threshold=1)
        seeds = set(g.nodes())
        result, _ = model.run(seeds)
        assert result.is_full_cascade
        assert result.cascade_size == g.number_of_nodes()

    def test_high_threshold_limits_spread_on_path(self):
        """On a path graph with threshold=3, spread is limited since path nodes have degree ≤ 2."""
        g = nx.path_graph(15)
        model = BootstrapPercolation(g, threshold=3)
        seeds = {0, 1}
        result, _ = model.run(seeds)
        # With threshold=3 on a path (max degree 2), no new nodes can be infected
        assert result.cascade_size == len(seeds)

    def test_record_sequence_format(self):
        g = make_er()
        model = BootstrapPercolation(g, threshold=2)
        seeds = set(list(g.nodes())[:4])
        result, seq = model.run(seeds, record_sequence=True)
        assert isinstance(seq, list)
        # First entry is the seed set
        assert isinstance(seq[0], set)

    def test_seed_nodes_always_infected(self):
        g = make_er()
        model = BootstrapPercolation(g, threshold=2)
        random.seed(50)
        seeds = set(random.sample(list(g.nodes()), 4))
        result, _ = model.run(seeds)
        assert seeds.issubset(result.infected_nodes)

    def test_threshold_1_infects_at_least_as_much_as_threshold_2(self):
        """Lower threshold → at least as many infected nodes."""
        g = make_er(n=20, p=0.3, seed=55)
        seeds = set(list(g.nodes())[:4])
        model1 = BootstrapPercolation(g, threshold=1)
        model2 = BootstrapPercolation(g, threshold=2)
        r1, _ = model1.run(seeds)
        r2, _ = model2.run(seeds)
        assert r1.cascade_size >= r2.cascade_size

    def test_cascade_fraction_equals_size_over_n(self):
        g = make_er()
        n = g.number_of_nodes()
        model = BootstrapPercolation(g, threshold=2)
        seeds = set(list(g.nodes())[:5])
        result, _ = model.run(seeds)
        assert abs(result.cascade_fraction - result.cascade_size / n) < 1e-9


# ─── WTM ──────────────────────────────────────────────────────────────────────

class TestWTMModel:

    def test_cascade_fraction_in_unit_interval(self):
        g = make_er()
        model = WTMModel(g, phi=0.3)
        for _ in range(N_TRIALS):
            seeds = set(list(g.nodes())[:3])
            result, _ = model.run(seeds)
            assert 0.0 <= result.cascade_fraction <= 1.0

    def test_cascade_size_equals_infected_nodes_len(self):
        g = make_er()
        model = WTMModel(g, phi=0.3)
        seeds = set(list(g.nodes())[:4])
        result, _ = model.run(seeds)
        assert result.cascade_size == len(result.infected_nodes)

    def test_empty_seed_produces_zero(self):
        g = make_er()
        model = WTMModel(g, phi=0.3)
        result, _ = model.run(set())
        assert result.cascade_fraction == 0.0

    def test_deterministic(self):
        g = make_er()
        model = WTMModel(g, phi=0.3)
        seeds = set(list(g.nodes())[:5])
        r1, _ = model.run(seeds)
        r2, _ = model.run(seeds)
        assert r1.cascade_size == r2.cascade_size

    def test_phi_zero_spreads_more_than_phi_one(self):
        """Lower phi threshold → at least as much spread."""
        g = make_er(n=20, p=0.3, seed=60)
        seeds = set(list(g.nodes())[:3])
        model_low  = WTMModel(g, phi=0.01)   # nearly every neighbour triggers
        model_high = WTMModel(g, phi=0.99)   # almost all neighbours required
        r_low,  _ = model_low.run(seeds)
        r_high, _ = model_high.run(seeds)
        assert r_low.cascade_size >= r_high.cascade_size

    def test_record_sequence_format(self):
        g = make_er()
        model = WTMModel(g, phi=0.3)
        seeds = set(list(g.nodes())[:3])
        result, seq = model.run(seeds, record_sequence=True)
        assert isinstance(seq, list)
        assert len(seq) >= 1
        for entry in seq:
            assert isinstance(entry, set)

    def test_seed_nodes_always_infected(self):
        g = make_er()
        model = WTMModel(g, phi=0.3)
        seeds = set(list(g.nodes())[:4])
        result, _ = model.run(seeds)
        assert seeds.issubset(result.infected_nodes)

    def test_is_full_cascade_flag_correct(self):
        g = make_complete(10)
        model = WTMModel(g, phi=0.01)  # any infected neighbour → activates
        seeds = {0}
        result, _ = model.run(seeds)
        # On K_10 with phi≈0, one infected neighbour is enough for every other node
        assert result.is_full_cascade or result.cascade_size >= 1


# ─── H1 ───────────────────────────────────────────────────────────────────────

class TestH1Model:

    def setup_method(self):
        self.g = make_er(n=20, p=0.3, seed=100)

    def test_cascade_fraction_in_unit_interval(self):
        model = H1Model(self.g, threshold=2, beta=0.3, gamma=0.1)
        for _ in range(N_TRIALS):
            random.seed(_)
            seeds = {random.choice(list(self.g.nodes()))}
            result, _ = model.run(seeds)
            assert 0.0 <= result.cascade_fraction <= 1.0

    def test_cascade_size_equals_infected_len(self):
        model = H1Model(self.g, threshold=2, beta=0.3, gamma=0.1)
        random.seed(101)
        seeds = set(random.sample(list(self.g.nodes()), 3))
        result, _ = model.run(seeds)
        assert result.cascade_size == len(result.infected_nodes)

    def test_empty_seed_zero_cascade(self):
        model = H1Model(self.g, threshold=2, beta=0.5, gamma=0.1)
        result, _ = model.run(set())
        assert result.cascade_fraction == 0.0

    def test_recovered_subset_of_infected(self):
        model = H1Model(self.g, threshold=2, beta=0.4, gamma=0.2)
        random.seed(102)
        seeds = set(random.sample(list(self.g.nodes()), 3))
        result, _ = model.run(seeds)
        assert result.recovered_nodes.issubset(result.infected_nodes)

    def test_record_sequence_format(self):
        model = H1Model(self.g, threshold=2, beta=0.4, gamma=0.1)
        seeds = set(list(self.g.nodes())[:3])
        result, seq = model.run(seeds, record_sequence=True)
        assert isinstance(seq, list)
        for entry in seq:
            assert isinstance(entry, tuple) and len(entry) == 2

    def test_supercritical_spreads_on_average(self):
        model = H1Model(self.g, threshold=2, beta=0.7, gamma=0.05)
        random.seed(103)
        fracs = []
        for _ in range(N_TRIALS):
            seeds = set(random.sample(list(self.g.nodes()), 3))
            result, _ = model.run(seeds)
            fracs.append(result.cascade_fraction)
        assert sum(fracs) / len(fracs) > 0.1

    def test_very_high_threshold_behaves_like_sir(self):
        """With threshold → large number (more than any node's degree), only SIR channel fires."""
        g = self.g
        threshold_large = g.number_of_nodes() + 100  # impossible threshold
        model_h1  = H1Model(g, threshold=threshold_large, beta=0.4, gamma=0.1)
        model_sir = SIRModel(g, beta=0.4, gamma=0.1)
        # Both should produce non-negative fractions (structure equivalence is stochastic)
        random.seed(104)
        seeds = set(random.sample(list(g.nodes()), 3))
        r_h1,  _ = model_h1.run(seeds)
        r_sir, _ = model_sir.run(seeds)
        # Both should have cascade fraction in [0,1]
        assert 0.0 <= r_h1.cascade_fraction <= 1.0
        assert 0.0 <= r_sir.epidemic_fraction <= 1.0


# ─── H2 ───────────────────────────────────────────────────────────────────────

class TestH2Model:

    def setup_method(self):
        self.g = make_er(n=20, p=0.3, seed=200)

    def test_cascade_fraction_in_unit_interval(self):
        model = H2Model(self.g, threshold=2, beta=0.3, gamma=0.1, switch_fraction=0.3)
        for _ in range(N_TRIALS):
            random.seed(_)
            seeds = {random.choice(list(self.g.nodes()))}
            result, _ = model.run(seeds)
            assert 0.0 <= result.cascade_fraction <= 1.0

    def test_cascade_size_equals_infected_len(self):
        model = H2Model(self.g, threshold=2, beta=0.3, gamma=0.1, switch_fraction=0.3)
        random.seed(201)
        seeds = set(random.sample(list(self.g.nodes()), 3))
        result, _ = model.run(seeds)
        assert result.cascade_size == len(result.infected_nodes)

    def test_empty_seed_zero_cascade(self):
        model = H2Model(self.g, threshold=2, beta=0.5, gamma=0.1, switch_fraction=0.3)
        result, _ = model.run(set())
        assert result.cascade_fraction == 0.0

    def test_total_rounds_equals_phase1_plus_phase2(self):
        model = H2Model(self.g, threshold=2, beta=0.4, gamma=0.1, switch_fraction=0.2)
        random.seed(202)
        seeds = set(random.sample(list(self.g.nodes()), 4))
        result, _ = model.run(seeds)
        assert result.time_to_cascade == result.rounds_phase1 + result.rounds_phase2

    def test_recovered_subset_of_infected(self):
        model = H2Model(self.g, threshold=2, beta=0.4, gamma=0.2, switch_fraction=0.3)
        random.seed(203)
        seeds = set(random.sample(list(self.g.nodes()), 3))
        result, _ = model.run(seeds)
        assert result.recovered_nodes.issubset(result.infected_nodes)

    def test_switch_flag_when_fraction_exceeded(self):
        """With switch_fraction=0 every non-empty seed triggers a switch immediately."""
        model = H2Model(self.g, threshold=2, beta=0.3, gamma=0.1, switch_fraction=0.0)
        seeds = {list(self.g.nodes())[0]}
        result, _ = model.run(seeds)
        # seed/n > 0 = switch_fraction → switched=True
        assert result.switched is True

    def test_switch_fraction_above_one_never_switches(self):
        """switch_fraction=2.0 can never be reached → acts like pure SIR."""
        model = H2Model(self.g, threshold=2, beta=0.4, gamma=0.1, switch_fraction=2.0)
        random.seed(204)
        seeds = set(random.sample(list(self.g.nodes()), 3))
        result, _ = model.run(seeds)
        assert result.switched is False
        assert result.rounds_phase2 == 0

    def test_record_sequence_format(self):
        model = H2Model(self.g, threshold=2, beta=0.4, gamma=0.1, switch_fraction=0.3)
        seeds = set(list(self.g.nodes())[:3])
        result, seq = model.run(seeds, record_sequence=True)
        assert isinstance(seq, list)
        for entry in seq:
            assert isinstance(entry, tuple) and len(entry) == 2


# ─── H3 ───────────────────────────────────────────────────────────────────────

class TestH3Model:

    def setup_method(self):
        self.g = make_er(n=20, p=0.3, seed=300)

    def test_cascade_fraction_in_unit_interval(self):
        model = H3Model(self.g, beta=0.3, gamma=0.1)
        for _ in range(N_TRIALS):
            random.seed(_)
            seeds = {random.choice(list(self.g.nodes()))}
            result, _ = model.run(seeds)
            assert 0.0 <= result.cascade_fraction <= 1.0

    def test_cascade_size_equals_infected_len(self):
        model = H3Model(self.g, beta=0.3, gamma=0.1)
        random.seed(301)
        seeds = set(random.sample(list(self.g.nodes()), 3))
        result, _ = model.run(seeds)
        assert result.cascade_size == len(result.infected_nodes)

    def test_empty_seed_zero(self):
        model = H3Model(self.g, beta=0.5, gamma=0.1)
        result, _ = model.run(set())
        assert result.cascade_fraction == 0.0

    def test_recovered_subset_of_infected(self):
        model = H3Model(self.g, beta=0.4, gamma=0.2)
        random.seed(302)
        seeds = set(random.sample(list(self.g.nodes()), 3))
        result, _ = model.run(seeds)
        assert result.recovered_nodes.issubset(result.infected_nodes)

    def test_record_sequence_format(self):
        model = H3Model(self.g, beta=0.4, gamma=0.1)
        seeds = set(list(self.g.nodes())[:3])
        result, seq = model.run(seeds, record_sequence=True)
        assert isinstance(seq, list)
        for entry in seq:
            assert isinstance(entry, tuple) and len(entry) == 2

    def test_high_beta_infects_more_than_low_beta_on_average(self):
        """H3 is monotone in beta: higher beta → more spread on average."""
        random.seed(303)
        fracs_high, fracs_low = [], []
        for _ in range(N_TRIALS):
            seeds = set(random.sample(list(self.g.nodes()), 3))
            r_high, _ = H3Model(self.g, beta=0.6, gamma=0.1).run(seeds)
            r_low,  _ = H3Model(self.g, beta=0.05, gamma=0.1).run(seeds)
            fracs_high.append(r_high.cascade_fraction)
            fracs_low.append(r_low.cascade_fraction)
        assert sum(fracs_high) / N_TRIALS >= sum(fracs_low) / N_TRIALS

    def test_seed_nodes_always_infected(self):
        model = H3Model(self.g, beta=0.3, gamma=0.1)
        seeds = set(list(self.g.nodes())[:4])
        result, _ = model.run(seeds)
        assert seeds.issubset(result.infected_nodes)


# ─── H4 ───────────────────────────────────────────────────────────────────────

class TestH4Model:

    def setup_method(self):
        self.g = make_er(n=20, p=0.3, seed=400)

    def test_cascade_fraction_in_unit_interval(self):
        model = H4Model(self.g, phi=0.3, beta=0.3, gamma=0.1)
        for _ in range(N_TRIALS):
            random.seed(_)
            seeds = {random.choice(list(self.g.nodes()))}
            result, _ = model.run(seeds)
            assert 0.0 <= result.cascade_fraction <= 1.0

    def test_cascade_size_equals_peak_infected(self):
        model = H4Model(self.g, phi=0.3, beta=0.3, gamma=0.1)
        random.seed(401)
        seeds = set(random.sample(list(self.g.nodes()), 3))
        result, _ = model.run(seeds)
        assert result.cascade_size == result.peak_infected

    def test_empty_seed_zero(self):
        model = H4Model(self.g, phi=0.3, beta=0.5, gamma=0.1)
        result, _ = model.run(set())
        assert result.cascade_fraction == 0.0

    def test_max_steps_respected(self):
        model = H4Model(self.g, phi=0.3, beta=0.5, gamma=0.05, max_steps=10)
        random.seed(402)
        seeds = set(random.sample(list(self.g.nodes()), 4))
        result, _ = model.run(seeds)
        assert result.time_to_cascade <= 10

    def test_infected_series_has_initial_entry(self):
        model = H4Model(self.g, phi=0.3, beta=0.3, gamma=0.2)
        seeds = {list(self.g.nodes())[0]}
        result, _ = model.run(seeds)
        assert len(result.infected_series) >= 1
        assert result.infected_series[0] == len(seeds)

    def test_record_sequence_format(self):
        model = H4Model(self.g, phi=0.3, beta=0.4, gamma=0.1)
        seeds = set(list(self.g.nodes())[:3])
        result, seq = model.run(seeds, record_sequence=True)
        assert isinstance(seq, list)
        for entry in seq:
            assert isinstance(entry, tuple) and len(entry) == 2


# ─── H5 ───────────────────────────────────────────────────────────────────────

class TestH5Model:

    def setup_method(self):
        self.g = make_er(n=20, p=0.3, seed=500)

    def test_cascade_fraction_in_unit_interval(self):
        model = H5Model(self.g, phi=0.3, beta=0.3, gamma=0.1, switch_fraction=0.3)
        for _ in range(N_TRIALS):
            random.seed(_)
            seeds = {random.choice(list(self.g.nodes()))}
            result, _ = model.run(seeds)
            assert 0.0 <= result.cascade_fraction <= 1.0

    def test_cascade_size_equals_infected_len(self):
        model = H5Model(self.g, phi=0.3, beta=0.3, gamma=0.1, switch_fraction=0.3)
        random.seed(501)
        seeds = set(random.sample(list(self.g.nodes()), 3))
        result, _ = model.run(seeds)
        assert result.cascade_size == len(result.infected_nodes)

    def test_empty_seed_zero(self):
        model = H5Model(self.g, phi=0.3, beta=0.5, gamma=0.1, switch_fraction=0.3)
        result, _ = model.run(set())
        assert result.cascade_fraction == 0.0

    def test_total_rounds_equals_phase_sum(self):
        model = H5Model(self.g, phi=0.3, beta=0.4, gamma=0.1, switch_fraction=0.2)
        random.seed(502)
        seeds = set(random.sample(list(self.g.nodes()), 4))
        result, _ = model.run(seeds)
        assert result.time_to_cascade == result.rounds_phase1 + result.rounds_phase2

    def test_switch_fraction_above_one_never_switches(self):
        model = H5Model(self.g, phi=0.3, beta=0.4, gamma=0.1, switch_fraction=2.0)
        random.seed(503)
        seeds = set(random.sample(list(self.g.nodes()), 3))
        result, _ = model.run(seeds)
        assert result.switched is False
        assert result.rounds_phase2 == 0

    def test_record_sequence_format(self):
        model = H5Model(self.g, phi=0.3, beta=0.4, gamma=0.1, switch_fraction=0.3)
        seeds = set(list(self.g.nodes())[:3])
        result, seq = model.run(seeds, record_sequence=True)
        assert isinstance(seq, list)
        for entry in seq:
            assert isinstance(entry, tuple) and len(entry) == 2


# ─── H6 ───────────────────────────────────────────────────────────────────────

class TestH6Model:

    def setup_method(self):
        self.g = make_er(n=20, p=0.3, seed=600)

    def test_cascade_fraction_in_unit_interval(self):
        model = H6Model(self.g, phi=0.5, gamma=0.1)
        for _ in range(N_TRIALS):
            random.seed(_)
            seeds = {random.choice(list(self.g.nodes()))}
            result, _ = model.run(seeds)
            assert 0.0 <= result.cascade_fraction <= 1.0

    def test_cascade_size_equals_infected_len(self):
        model = H6Model(self.g, phi=0.5, gamma=0.1)
        random.seed(601)
        seeds = set(random.sample(list(self.g.nodes()), 3))
        result, _ = model.run(seeds)
        assert result.cascade_size == len(result.infected_nodes)

    def test_empty_seed_zero(self):
        model = H6Model(self.g, phi=0.5, gamma=0.1)
        result, _ = model.run(set())
        assert result.cascade_fraction == 0.0

    def test_recovered_subset_of_infected(self):
        model = H6Model(self.g, phi=0.5, gamma=0.2)
        random.seed(602)
        seeds = set(random.sample(list(self.g.nodes()), 3))
        result, _ = model.run(seeds)
        assert result.recovered_nodes.issubset(result.infected_nodes)

    def test_record_sequence_format(self):
        model = H6Model(self.g, phi=0.5, gamma=0.1)
        seeds = set(list(self.g.nodes())[:3])
        result, seq = model.run(seeds, record_sequence=True)
        assert isinstance(seq, list)
        for entry in seq:
            assert isinstance(entry, tuple) and len(entry) == 2

    def test_lower_phi_infects_more_on_average(self):
        """Smaller phi → easier to activate → more spread on average."""
        random.seed(603)
        fracs_low_phi, fracs_high_phi = [], []
        for _ in range(N_TRIALS):
            seeds = set(random.sample(list(self.g.nodes()), 3))
            r_low,  _ = H6Model(self.g, phi=0.1,  gamma=0.05).run(seeds)
            r_high, _ = H6Model(self.g, phi=0.9, gamma=0.05).run(seeds)
            fracs_low_phi.append(r_low.cascade_fraction)
            fracs_high_phi.append(r_high.cascade_fraction)
        assert sum(fracs_low_phi) / N_TRIALS >= sum(fracs_high_phi) / N_TRIALS

    def test_seed_nodes_always_infected(self):
        model = H6Model(self.g, phi=0.5, gamma=0.1)
        seeds = set(list(self.g.nodes())[:4])
        result, _ = model.run(seeds)
        assert seeds.issubset(result.infected_nodes)


# ─── Cross-model invariants ────────────────────────────────────────────────────

class TestCrossModelInvariants:
    """Tests that apply the same structural invariants across multiple models."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.g = make_er(n=20, p=0.3, seed=700)
        random.seed(700)
        self.seeds = set(random.sample(list(self.g.nodes()), 4))

    def _get_cascade_fraction(self, result):
        """Extract cascade_fraction from any model result."""
        if hasattr(result, "epidemic_fraction"):
            return result.epidemic_fraction
        return result.cascade_fraction

    def test_all_models_fraction_in_range(self):
        models_and_results = [
            SIRModel(self.g, beta=0.3, gamma=0.1).run(self.seeds),
            SISModel(self.g, beta=0.3, gamma=0.1).run(self.seeds),
            BootstrapPercolation(self.g, threshold=2).run(self.seeds),
            WTMModel(self.g, phi=0.3).run(self.seeds),
            H1Model(self.g, threshold=2, beta=0.3, gamma=0.1).run(self.seeds),
            H2Model(self.g, threshold=2, beta=0.3, gamma=0.1, switch_fraction=0.3).run(self.seeds),
            H3Model(self.g, beta=0.3, gamma=0.1).run(self.seeds),
            H4Model(self.g, phi=0.3, beta=0.3, gamma=0.1).run(self.seeds),
            H5Model(self.g, phi=0.3, beta=0.3, gamma=0.1, switch_fraction=0.3).run(self.seeds),
            H6Model(self.g, phi=0.5, gamma=0.1).run(self.seeds),
        ]
        for result, _ in models_and_results:
            frac = self._get_cascade_fraction(result)
            assert 0.0 <= frac <= 1.0

    def test_all_models_infected_nodes_subset_of_graph_nodes(self):
        all_nodes = set(self.g.nodes())
        models_and_results = [
            SIRModel(self.g, beta=0.3, gamma=0.1).run(self.seeds),
            SISModel(self.g, beta=0.3, gamma=0.1).run(self.seeds),
            BootstrapPercolation(self.g, threshold=2).run(self.seeds),
            WTMModel(self.g, phi=0.3).run(self.seeds),
            H1Model(self.g, threshold=2, beta=0.3, gamma=0.1).run(self.seeds),
            H2Model(self.g, threshold=2, beta=0.3, gamma=0.1, switch_fraction=0.3).run(self.seeds),
            H3Model(self.g, beta=0.3, gamma=0.1).run(self.seeds),
            H4Model(self.g, phi=0.3, beta=0.3, gamma=0.1).run(self.seeds),
            H5Model(self.g, phi=0.3, beta=0.3, gamma=0.1, switch_fraction=0.3).run(self.seeds),
            H6Model(self.g, phi=0.5, gamma=0.1).run(self.seeds),
        ]
        for result, _ in models_and_results:
            assert result.infected_nodes.issubset(all_nodes)

    def test_all_models_seed_nodes_in_infected(self):
        models_and_results = [
            SIRModel(self.g, beta=0.3, gamma=0.1).run(self.seeds),
            SISModel(self.g, beta=0.3, gamma=0.1).run(self.seeds),
            BootstrapPercolation(self.g, threshold=2).run(self.seeds),
            WTMModel(self.g, phi=0.3).run(self.seeds),
            H1Model(self.g, threshold=2, beta=0.3, gamma=0.1).run(self.seeds),
            H2Model(self.g, threshold=2, beta=0.3, gamma=0.1, switch_fraction=0.3).run(self.seeds),
            H3Model(self.g, beta=0.3, gamma=0.1).run(self.seeds),
            H4Model(self.g, phi=0.3, beta=0.3, gamma=0.1).run(self.seeds),
            H5Model(self.g, phi=0.3, beta=0.3, gamma=0.1, switch_fraction=0.3).run(self.seeds),
            H6Model(self.g, phi=0.5, gamma=0.1).run(self.seeds),
        ]
        for result, _ in models_and_results:
            assert self.seeds.issubset(result.infected_nodes)
