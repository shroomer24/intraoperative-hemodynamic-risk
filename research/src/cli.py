"""A configuration smoke check; deliberately no training command."""

import argparse
from pathlib import Path

from intraop.config import load_config
from intraop.logging import configure_logging
from intraop.reproducibility import seed_everything, write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check-config"])
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, help="Optional runtime manifest output path")
    args = parser.parse_args()
    config = load_config(args.config, project_root=args.project_root)
    logger = configure_logging(config.log_level)
    seed_everything(config.seed)
    if args.manifest:
        write_run_manifest(args.manifest, config)
    logger.info(
        "Configuration validated. VitalDB v0.1 is specified; model training remains unimplemented."
    )


if __name__ == "__main__":
    main()
