"""Hosted candidate worker: launch only through the normal-terminal wrapper."""

from modeling_candidate_worker import main

if __name__ == "__main__":
    raise SystemExit(main("tabpfn"))
