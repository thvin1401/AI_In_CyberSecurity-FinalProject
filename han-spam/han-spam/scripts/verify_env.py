#!/usr/bin/env python
"""
scripts/verify_env.py
Run inside the container to confirm the Phase 1 environment is healthy.

Usage:
    docker compose run --rm train python scripts/verify_env.py
"""
import sys

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def check(label: str, fn):
    try:
        result = fn()
        print(f"  {PASS}  {label}: {result}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {label}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n══════════════════════════════════════════")
    print("  HAN-Spam  |  Phase 1 environment check")
    print("══════════════════════════════════════════\n")

    results = []

    # ── Python ──────────────────────────────────
    results.append(check("Python version", lambda: sys.version.split()[0]))

    # ── TensorFlow / GPU ────────────────────────
    def tf_check():
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        return f"v{tf.__version__}, GPUs detected: {len(gpus)}"

    results.append(check("TensorFlow + GPU", tf_check))

    # ── Keras ───────────────────────────────────
    def keras_check():
        import keras
        return f"v{keras.__version__}"

    results.append(check("Keras", keras_check))

    # ── NLTK data ───────────────────────────────
    def nltk_check():
        import nltk
        tokens = nltk.sent_tokenize("Hello world. This is a test.")
        assert len(tokens) == 2
        return f"punkt OK, tokenised {len(tokens)} sentences"

    results.append(check("NLTK sent_tokenize", nltk_check))

    # ── FastText ────────────────────────────────
    def ft_check():
        import fasttext
        assert hasattr(fasttext, "train_unsupervised")
        return "import OK, train_unsupervised available"

    results.append(check("FastText", ft_check))

    # ── scikit-learn ────────────────────────────
    def sklearn_check():
        import sklearn
        return f"v{sklearn.__version__}"

    results.append(check("scikit-learn", sklearn_check))

    # ── MLflow ──────────────────────────────────
    def mlflow_check():
        import mlflow
        return f"v{mlflow.__version__}"

    results.append(check("MLflow", mlflow_check))

    # ── Config loader ───────────────────────────
    def config_check():
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from src.utils.config import CFG
        return f"project={CFG.project.name}, seed={CFG.project.seed}"

    results.append(check("Config loader (src.utils.config)", config_check))

    # ── Seed util ───────────────────────────────
    def seed_check():
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from src.utils.seed import set_seed
        set_seed(42)
        return "OK"

    results.append(check("Seed utility", seed_check))

    # ── Summary ─────────────────────────────────
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"\n  {n_pass}/{len(results)} checks passed", end="")
    if n_fail:
        print(f"  ({n_fail} FAILED — fix before proceeding to Phase 2)")
        sys.exit(1)
    else:
        print("  — environment is ready for Phase 2 ✓")


if __name__ == "__main__":
    main()
