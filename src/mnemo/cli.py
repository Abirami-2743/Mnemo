"""`mnemo` command line: train | evaluate | serve."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="mnemo", description="MNEMO memory-management toolkit")
    ap.add_argument("command", choices=["train", "evaluate", "serve"])
    args, rest = ap.parse_known_args(argv)
    if args.command == "train":
        from .rl.train import main as run

        run(rest)
    elif args.command == "evaluate":
        from .run_eval import main as run

        run(rest)
    else:
        import uvicorn

        sp = argparse.ArgumentParser(prog="mnemo serve")
        sp.add_argument("--host", default="127.0.0.1")
        sp.add_argument("--port", type=int, default=8000)
        sa = sp.parse_args(rest)
        uvicorn.run("mnemo.service.app:app_factory", factory=True, host=sa.host, port=sa.port)


if __name__ == "__main__":
    main()
