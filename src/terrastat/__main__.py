"""Lets ``python -m terrastat ...`` work as well as the installed ``terrastat`` command.

Both run the same CLI. The module form is useful when the virtualenv's Scripts directory is not
on PATH, which on Windows is the common case: ``.venv\\Scripts\\python -m terrastat run``.
"""
from terrastat.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
