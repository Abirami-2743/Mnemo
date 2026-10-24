"""Export a trained checkpoint to a torch-free .npz: python scripts/export_policy.py [model.zip] [out.npz]"""

import sys
from pathlib import Path

from mnemo.rl.numpy_policy import export_policy

root = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "models" / "mnemo_ppo" / "best_model.zip"
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name("policy.npz")
print("wrote", export_policy(src, dst))
