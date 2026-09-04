from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LNP.LatentNP import LatNP
from TRL.trl_training import train_trl_model


def main() -> None:
    train_trl_model(LatNP, "ANP")


if __name__ == "__main__":
    main()
