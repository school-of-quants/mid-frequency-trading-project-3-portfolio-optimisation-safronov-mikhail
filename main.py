import warnings
import sys
import os
from pathlib import Path

current_file = Path(__file__).resolve()

project_path = current_file.parent.parent 

if str(project_path.parent) not in sys.path:
    sys.path.insert(0, str(project_path.parent))


from equity_project.src.get_data import get_data
from equity_project.src.run_backtest import run_backtest
from equity_project.src.train import train


def main():
    get_data()
    train()
    run_backtest()


if __name__ == "__main__":
    main()
