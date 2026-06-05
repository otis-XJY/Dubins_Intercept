import numpy as np
import os
import sys
import importlib.util


def main():
    # Minimal deterministic validation for stable-length capture/distance state.
    # This script does NOT rely on a short episode randomly producing captures.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    # Avoid importing the `marl` package top-level (it may require torch).
    env_path = os.path.join(repo_root, "marl", "envs", "todc_env.py")
    spec = importlib.util.spec_from_file_location("todc_env_module", env_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {env_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    TODCMARLEnv = getattr(mod, "TODCMARLEnv")

    env = TODCMARLEnv({"debug_print": False, "render_mode": "none"})
    env.reset(seed=0)

    # Make capture condition easy to satisfy: large distance, full angle.
    env.CapRef["CapDist"] = float(1e6)
    env.CapRef["CapAngle"] = float(360.0)

    # Force compact alive sets to include two pursuers/enemies (use global ids 0 and 1).
    if env.num_P < 2 or env.num_E < 2:
        raise RuntimeError(f"Need at least 2 pursuers/enemies, got num_P={env.num_P}, num_E={env.num_E}")

    env.UnCapPidNew = np.array([0, 1], dtype=int)
    env.UnCapEidNew = np.array([0, 1], dtype=int)

    # Provide compact positions consistent with UnCap*New ordering.
    # Step 1: put pursuers close enough to capture eid=0 only, eid=1 far.
    env.PosP = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]], dtype=float)
    env.PosE = np.array([[1.0, 0.0, 0.0], [1e8, 0.0, 0.0]], dtype=float)

    env.Capflag_full[:] = False
    env.assigned_eid_full[:] = -1
    env.assigned_eid_full[0] = 0
    env.assigned_eid_full[1] = 1
    env.last_min_dist_stable = np.full((env.num_P,), np.inf, dtype=np.float32)

    newly1 = env._update_capflag_full_from_geometry()
    if int(np.sum(env.Capflag_full)) != 1 or newly1 != 1:
        raise AssertionError(f"Expected captured_total_full=1 newly=1, got sum={np.sum(env.Capflag_full)} newly={newly1}")

    curr1 = env._compute_curr_min_dist_stable()
    if curr1.shape != (env.num_P,):
        raise AssertionError(f"curr_min_dist_stable shape wrong: {curr1.shape}")
    if not np.isfinite(curr1[1]) and env.assigned_eid_full[1] == 1 and not env.Capflag_full[1]:
        raise AssertionError("Expected finite distance for pid=1 to eid=1 before second capture")

    env.last_min_dist_stable = curr1.copy()

    # Step 2: move eid=1 near pid=1 to force second capture.
    env.PosE = np.array([[1.0, 0.0, 0.0], [1001.0, 0.0, 0.0]], dtype=float)
    newly2 = env._update_capflag_full_from_geometry()
    if int(np.sum(env.Capflag_full)) != 2 or newly2 != 1:
        raise AssertionError(f"Expected captured_total_full=2 newly=1, got sum={np.sum(env.Capflag_full)} newly={newly2}")

    curr2 = env._compute_curr_min_dist_stable()
    if curr2.shape != (env.num_P,):
        raise AssertionError(f"curr_min_dist_stable shape wrong after capture: {curr2.shape}")

    print("[OK] forced capture delta: 0->1->2 and stable dist shape=(num_P,) verified.")


if __name__ == "__main__":
    main()

