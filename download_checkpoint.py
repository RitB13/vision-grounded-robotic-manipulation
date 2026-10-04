from pathlib import Path
import gdown

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "checkpoints" / "ckpt_40000"
OUT.parent.mkdir(exist_ok=True)

if OUT.exists():
    print(f"Checkpoint already exists: {OUT}")
else:
    print("Downloading pretrained CLIPort checkpoint...")
    gdown.download(
        id="1Nq0q1KbqHOA5O7aRSu4u7-u27EMMXqgP",
        output=str(OUT),
        quiet=False,
    )

print("Checkpoint:", OUT)
