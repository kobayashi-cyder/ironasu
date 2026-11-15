"""Frozen application entry point. PyInstaller includes Python and bundled UI assets."""
from ironasu.studio import main

if __name__ == "__main__":
    raise SystemExit(main())
