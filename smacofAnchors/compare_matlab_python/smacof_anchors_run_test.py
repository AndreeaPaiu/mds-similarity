import json
import numpy as np
import smacofAnchors.smacofAnchors2026 as  sma
# --- Main Logic to Fill JSON ---
def fill_python_results(filename="mds_100_tests.json"):
    with open(filename, 'r') as f:
        tests = json.load(f)

    for test in tests:
        # Extract inputs from JSON
        X_targets_real = np.array(test['inputs']['X_targets_real'])
        X_anchors = np.array(test['inputs']['X_anchors'])
        X0 = np.array(test['inputs']['X0'])
        w_tt = test['inputs']['w_tt']
        w_ta = test['inputs']['w_ta']
        n_anchors = test['config']['n_anchors']
        n_targets = test['config']['n_targets']
        nodes = n_targets + n_anchors

        # 1. Create Ground Truth Distance Matrix (D)
        X_real = np.vstack([X_targets_real, X_anchors])
        D = sma.compute_D(X_real)

        # 2. Create Weight Matrix (W)
        W = np.ones((nodes, nodes))
        W[0:n_targets, 0:n_targets] = w_tt
        W[0:n_targets, n_targets:] = w_ta
        W[n_targets:, 0:n_targets] = w_ta
        W[n_targets:, n_targets:] = 0  # Anchor-Anchor fixed
        np.fill_diagonal(W, 0)

        # 3. Run SMACOF
        X_est, final_stress = sma.smacof_anchors(D, X0, n_anchors, W, sma.MDSParam())

        # 4. Fill results back into JSON structure
        test['results']['python_X_est'] = X_est.tolist()
        test['results']['python_stress'] = float(final_stress)

    # Save updated file
    with open(filename, 'w') as f:
        json.dump(tests, f, indent=4)
    print(f"Successfully filled Python results for {len(tests)} tests.")


if __name__ == "__main__":
    fill_python_results()