"""
Run all undersampling training scripts in parallel.

Each script undersamples once then tunes all classifiers, so they are
independent and safe to run concurrently. Adjust MAX_WORKERS to match
available CPU cores — note that each individual script already uses
n_jobs=-1 internally, so high parallelism here will cause contention.
"""
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

MAX_WORKERS = 4

SCRIPTS = [
    "train-rus.py",
    "train-cnn.py",
    "train-tomek.py",
    "train-oss.py",
    "train-enn.py",
    "train-renn.py",
    "train-allknn.py",
    "train-ncr.py",
    "train-nm1.py",
    "train-nm2.py",
]

SCRIPTS_DIR = Path(__file__).resolve().parent


def run_script(script):
    result = subprocess.run(
        [sys.executable, SCRIPTS_DIR / script],
        capture_output=True,
        text=True,
    )
    return script, result


if __name__ == "__main__":
    print(f"Running {len(SCRIPTS)} scripts with MAX_WORKERS={MAX_WORKERS}\n")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_script, s): s for s in SCRIPTS}

        for future in as_completed(futures):
            script, result = future.result()
            status = "OK" if result.returncode == 0 else "FAILED"
            print(f"[{status}] {script}")
            if result.returncode != 0:
                print(result.stderr)
