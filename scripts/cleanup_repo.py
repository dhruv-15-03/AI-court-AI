import os
from glob import glob
import shutil
from typing import List


def safe_unlink(path: str):
    try:
        if os.path.isfile(path):
            os.remove(path)
            print(f"Removed: {path}")
    except Exception as e:
        print(f"Skip {path}: {e}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    # Delete debug artifacts
    for pattern in [
        "debug_*.html",
        "debug_*.txt",
        "*.log",
        "simple_scraper.log",
        "scraper.log",
        "legal_scraper.log",
    ]:
        for p in glob(pattern):
            safe_unlink(p)

    # Ensure models directory exists
    models_dir = os.path.join(root, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Pick the newest classifier artifact in root and move it into models/
    artifacts: List[str] = sorted(glob("legal_case_classifier*.pkl"), key=lambda p: os.path.getmtime(p), reverse=True)
    if artifacts:
        newest = artifacts[0]
        dest = os.path.join(models_dir, "legal_case_classifier.pkl")
        try:
            shutil.move(newest, dest)
            print(f"Moved canonical model to: {dest}")
        except Exception as e:
            print(f"Could not move {newest} -> {dest}: {e}")
        # Remove any remaining duplicates
        for p in artifacts[1:]:
            safe_unlink(p)

    # Remove stray label encoder artifacts
    for p in glob("label_encoder.pkl"):
        safe_unlink(p)

    print("Cleanup complete. Fresh artifacts will be saved under models/ and logs/.")


if __name__ == "__main__":
    main()
