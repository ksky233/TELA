"""Train and reload a tiny synthetic case in a temporary directory on CPU."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.synthetic import write_synthetic_csv  # noqa: E402


def main():
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with tempfile.TemporaryDirectory(prefix="tela_smoke_") as directory:
        work = Path(directory)
        write_synthetic_csv(work / "synthetic.csv")
        common = [
            sys.executable, str(ROOT / "run.py"),
            "--exp-name", "smoke", "--turned", "seed2026",
            "--root-path", str(work), "--data-path", "synthetic.csv",
            "--checkpoints", str(work / "checkpoints"),
            "--seq-len", "16", "--label-len", "8", "--pred-len", "4",
            "--patch-len", "4", "--patch-stride", "4",
            "--d-model", "16", "--d-ff", "32", "--n-heads", "2", "--e-layers", "1",
            "--top-k", "4", "--batch-size", "16", "--train-epochs", "1", "--patience", "1",
            "--no-use-gpu", "--no-use-revin", "--wind-dir-encoding", "sincos",
        ]
        for mode, folder in [("1", "train"), ("0", "evaluation")]:
            command = common + ["--is-training", mode, "--result-path", str(work / folder)]
            # Run outside the repository to also test path independence.
            subprocess.run(command, cwd=work, env=env, check=True)
        results = [json.loads((work / folder / "smoke" / "turnd-seed2026.json").read_text(encoding="utf-8"))
                   for folder in ("train", "evaluation")]
        if results[0]["metrics"] != results[1]["metrics"]:
            raise RuntimeError("Train/reload predictions produced different metrics")
        if results[0]["metrics"]["pred_shape"] != [61, 4, 1]:
            raise RuntimeError("Unexpected synthetic test shape")
        print("PASS: CPU training, checkpoint reload, exact metric agreement, portable paths.")
        print("Synthetic inputs were used only for software verification; temporary outputs removed.")


if __name__ == "__main__":
    main()
