import json
import numpy as np


def generate_bulk_tests(filename="mds_100_tests.json", num_tests=100):
    all_tests = []

    for i in range(num_tests):
        # Randomly choose number of targets (3 to 10) and anchors (3 to 8)
        n_targets = np.random.randint(3, 11)
        n_anchors = np.random.randint(3, 9)
        dim = 3

        # 1. Generate Ground Truth (Real Positions)
        # Targets are usually in a smaller central area, anchors spread out
        targets_real = np.random.rand(n_targets, dim) * 5
        anchors_real = np.random.rand(n_anchors, dim) * 10

        # 2. Generate Initial Guess (X0)
        # In SMACOF with anchors, the anchor part of X0 MUST match the real anchors
        X0_targets = np.random.rand(n_targets, dim) * 10
        X0 = np.vstack([X0_targets, anchors_real])

        # 3. Define weights (Using your previous logic)
        # These can be fixed or randomized slightly per test if desired
        w_tt = 0.2
        w_ta = 0.7

        test_case = {
            "test_id": i + 1,
            "config": {
                "n_targets": n_targets,
                "n_anchors": n_anchors,
                "dim": dim
            },
            "inputs": {
                "X_targets_real": targets_real.tolist(),
                "X_anchors": anchors_real.tolist(),
                "X0": X0.tolist(),
                "w_tt": w_tt,
                "w_ta": w_ta
            },
            "results": {
                "matlab_X_est": None,
                "matlab_stress": None,
                "python_X_est": None,
                "python_stress": None
            }
        }
        all_tests.append(test_case)

    with open(filename, 'w') as f:
        json.dump(all_tests, f, indent=4)

    print(f"Generated {num_tests} test cases in {filename}")


generate_bulk_tests()