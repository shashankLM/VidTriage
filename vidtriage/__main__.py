from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.wayland.textinput=false")

from PySide6.QtWidgets import QApplication

from .wizard import SetupWizard
from .session import Session
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

    wizard = SetupWizard(prefill_input=args.input_dir, prefill_output=args.output_dir)
    if wizard.exec() != wizard.DialogCode.Accepted or wizard.result_config is None:
        sys.exit(0)

    rc = wizard.result_config
    session = Session(rc.input_dir, rc.output_dir, rc.classes)
    session.load()

    window = MainWindow(session)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
