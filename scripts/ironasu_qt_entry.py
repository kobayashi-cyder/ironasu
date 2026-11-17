"""Frozen application entry with a private, windowless HTTP worker."""
import sys

if "--ironasu-model-worker" in sys.argv[1:2]:
    from ironasu.model_worker import main
else:
    from ironasu.qt_desktop import main

if __name__ == "__main__":
    raise SystemExit(main())
