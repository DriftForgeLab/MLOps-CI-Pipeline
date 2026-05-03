# =============================================================================
# src/monitoring/rollback_drift_cli.py — rollback-drift-training CLI
# =============================================================================
# Cleans up files added to the raw dataset by a prior `prepare-drift-training`
# invocation. Useful when a drift-adaptive fine-tuning experiment is rejected
# at promotion time (or aborted) and the raw dataset should be restored.
#
# Each `prepare-drift-training` run writes a manifest under
# data/raw/<dataset>/.drift_attempts/<timestamp>.json listing every file it
# copied. This CLI reads one or more of those manifests and unlinks exactly
# those files. Files outside the manifest are never touched.
#
# Usage:
#   rollback-drift-training --config src/config/pipeline_cifar10.yaml --list
#   rollback-drift-training --config src/config/pipeline_cifar10.yaml --latest
#   rollback-drift-training --config src/config/pipeline_cifar10.yaml --attempt 20260428_091200
#   rollback-drift-training --config src/config/pipeline_cifar10.yaml --all --yes
#   rollback-drift-training --config src/config/pipeline_cifar10.yaml --latest --remove-holdout
# =============================================================================

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Roll back files added to the raw dataset by a prior "
            "prepare-drift-training run. Reads the per-attempt manifest under "
            "data/raw/<dataset>/.drift_attempts/ and deletes only the files "
            "recorded there."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Pipeline config YAML (used to resolve dataset name and raw dir).",
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--list",
        action="store_true",
        help="List drift attempts and exit without modifying anything.",
    )
    selector.add_argument(
        "--latest",
        action="store_true",
        help="Roll back the most recent drift attempt.",
    )
    selector.add_argument(
        "--attempt",
        type=str,
        default=None,
        help="Roll back a specific attempt by its id (timestamp like 20260428_091200).",
    )
    selector.add_argument(
        "--all",
        action="store_true",
        help="Roll back every recorded drift attempt.",
    )
    parser.add_argument(
        "--remove-holdout",
        action="store_true",
        help=(
            "Also recursively delete the holdout directory recorded in the "
            "manifest. Off by default so historical baseline metrics survive."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt (useful for scripts).",
    )
    return parser.parse_args()


def _confirm(prompt: str) -> bool:
    try:
        answer = input(prompt).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    return answer in ("y", "yes")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()

    from src.config.loader import load_config
    from src.data.drift_adaptation import (
        list_drift_attempts,
        rollback_drift_attempt,
    )

    try:
        config = load_config(Path(args.config).resolve())
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load pipeline config: %s", exc)
        sys.exit(1)

    raw_dataset_dir = Path(config.data.raw) / config.dataset
    attempts = list_drift_attempts(raw_dataset_dir)

    if not attempts:
        print(f"No drift attempts recorded under {raw_dataset_dir / '.drift_attempts'}.")
        sys.exit(0)

    if args.list:
        _print_attempts(attempts)
        sys.exit(0)

    # Resolve which manifest(s) to roll back
    if args.all:
        targets = attempts
    elif args.latest:
        targets = [attempts[-1]]
    else:
        # --attempt
        matches = [a for a in attempts if a.get("attempt_id") == args.attempt]
        if not matches:
            available = ", ".join(a.get("attempt_id", "?") for a in attempts)
            logger.error(
                "No attempt found with id '%s'. Available: %s", args.attempt, available
            )
            sys.exit(1)
        targets = matches

    _print_attempts(targets, header="Will roll back:")
    if args.remove_holdout:
        print("  (--remove-holdout: holdout directories will also be deleted)")
    print()

    if not args.yes:
        ok = _confirm(f"Proceed with rollback of {len(targets)} attempt(s)? [y/N]: ")
        if not ok:
            print("Aborted.")
            sys.exit(0)

    total_removed = 0
    total_missing = 0
    for attempt in targets:
        result = rollback_drift_attempt(
            Path(attempt["manifest_path"]),
            remove_holdout=args.remove_holdout,
        )
        total_removed += result["removed"]
        total_missing += result["missing"]

    print()
    print("=" * 62)
    print("  ROLLBACK COMPLETE")
    print("=" * 62)
    print(f"  Attempts rolled back: {len(targets)}")
    print(f"  Files removed:        {total_removed}")
    if total_missing:
        print(f"  Files already gone:   {total_missing}")
    print()


def _print_attempts(attempts: list[dict], header: str = "Drift attempts:") -> None:
    print(header)
    for a in attempts:
        attempt_id = a.get("attempt_id", "?")
        created = a.get("created_at", "?")
        n_files = a.get("n_files_added", 0)
        drifted_dir = a.get("drifted_dir", "?")
        baseline = a.get("baseline_metrics") or {}
        acc = baseline.get("accuracy")
        acc_str = f"  baseline_acc={acc:.4f}" if isinstance(acc, (int, float)) else ""
        print(f"  - {attempt_id}  ({created})  n={n_files}{acc_str}")
        print(f"      from: {drifted_dir}")


if __name__ == "__main__":
    main()
