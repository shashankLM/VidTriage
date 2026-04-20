from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .config import load_config
from .wizard import SetupWizard
from .main_window import MainWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vidtriage",
        description="Rapidly classify videos into user-defined categories using keyboard shortcuts.",
    )
    parser.add_argument(
        "-i", "--input", dest="input_dir", type=Path, default=None,
        help="Input directory containing videos to classify",
    )
    parser.add_argument(
        "-o", "--output", dest="output_dir", type=Path, default=None,
        help="Output directory for classified videos",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("VidTriage")

    saved = load_config()
    prefill_input = args.input_dir or saved.input_dir
    prefill_output = args.output_dir or saved.output_dir

    wizard = SetupWizard(prefill_input=prefill_input, prefill_output=prefill_output)
    if wizard.exec() != wizard.DialogCode.Accepted or wizard.result_config is None:
        sys.exit(0)

    window = MainWindow(wizard.result_config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
