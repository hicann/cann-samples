#!/usr/bin/env python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Minimal manifest-driven CI functional test runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CommandResult:
    name: str
    phase: str
    cmd: list[str]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Manifest root must be a mapping.")
    return data


def ensure_path_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def run_command(entry: dict[str, Any], phase: str, artifact_dir: Path | None) -> CommandResult:
    name = entry["name"]
    cmd = entry["cmd"]
    cwd = REPO_ROOT / entry["cwd"]

    ensure_path_exists(cwd, f"Working directory for {phase}:{name}")
    if not isinstance(cmd, list) or not cmd:
        raise ValueError(f"{phase}:{name} cmd must be a non-empty argv list.")

    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    result = CommandResult(
        name=name,
        phase=phase,
        cmd=[str(part) for part in cmd],
        cwd=str(cwd.relative_to(REPO_ROOT)),
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{phase}__{name}"
        (artifact_dir / f"{stem}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (artifact_dir / f"{stem}.stderr.log").write_text(result.stderr, encoding="utf-8")
        (artifact_dir / f"{stem}.exit_code").write_text(f"{result.exit_code}\n", encoding="utf-8")

    return result


def check_contains(text: str, patterns: list[str], negate: bool) -> list[str]:
    failures: list[str] = []
    for pattern in patterns:
        exists = pattern in text
        if negate and exists:
            failures.append(f'unexpected pattern present: "{pattern}"')
        if not negate and not exists:
            failures.append(f'missing expected pattern: "{pattern}"')
    return failures


def evaluate_pass_criteria(sample: dict[str, Any], results: list[CommandResult]) -> list[str]:
    failures: list[str] = []
    criteria = sample.get("pass_criteria", {})
    if not results:
        return ["no commands were executed"]

    final_result = results[-1]

    expected_exit = criteria.get("exit_code")
    if expected_exit is not None and final_result.exit_code != expected_exit:
        failures.append(
            f"final exit code mismatch: expected {expected_exit}, got {final_result.exit_code}"
        )

    if criteria.get("all_steps_exit_zero"):
        for result in results:
            if result.exit_code != 0:
                failures.append(
                    f"{result.phase}:{result.name} exit code is {result.exit_code}, expected 0"
                )

    failures.extend(check_contains(final_result.stdout, criteria.get("stdout_contains", []), negate=False))
    failures.extend(check_contains(final_result.stdout, criteria.get("stdout_not_contains", []), negate=True))
    failures.extend(check_contains(final_result.stdout, criteria.get("final_stdout_contains", []), negate=False))
    return failures


def select_samples(manifest: dict[str, Any], sample_ids: set[str] | None) -> list[dict[str, Any]]:
    samples = manifest.get("samples", [])
    if sample_ids:
        available_ids = {sample["id"] for sample in samples}
        unknown_ids = sorted(sample_ids - available_ids)
        if unknown_ids:
            unknown = ", ".join(unknown_ids)
            raise ValueError(f"unknown sample id(s): {unknown}")
    return [sample for sample in samples if not sample_ids or sample["id"] in sample_ids]


def run_sample(sample: dict[str, Any], artifact_root: Path | None) -> dict[str, Any]:
    sample_id = sample["id"]
    sample_artifact_dir = artifact_root / sample_id if artifact_root is not None else None

    results: list[CommandResult] = []
    try:
        for entry in sample.get("setup", []):
            results.append(run_command(entry, "setup", sample_artifact_dir))
        for entry in sample.get("steps", []):
            results.append(run_command(entry, "step", sample_artifact_dir))
    except Exception as exc:
        return {
            "id": sample_id,
            "status": "failed",
            "errors": [str(exc)],
            "results": [result.__dict__ for result in results],
        }

    failures = evaluate_pass_criteria(sample, results)
    return {
        "id": sample_id,
        "status": "passed" if not failures else "failed",
        "errors": failures,
        "results": [result.__dict__ for result in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run manifest-driven CI functional tests.")
    parser.add_argument(
        "--manifest",
        default="tests/ci_functional_test.yaml",
        help="Path to the manifest file relative to the repository root.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Run only the specified sample id. Can be repeated.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="test_artifacts/ci_functional",
        help="Directory for stdout/stderr/exit code artifacts, relative to the repository root.",
    )
    args = parser.parse_args()

    manifest_path = REPO_ROOT / args.manifest
    ensure_path_exists(manifest_path, "Manifest")
    manifest = load_manifest(manifest_path)

    artifact_root = REPO_ROOT / args.artifacts_dir if args.artifacts_dir else None
    if artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)

    sample_ids = set(args.sample) if args.sample else None
    try:
        samples = select_samples(manifest, sample_ids)
    except ValueError as exc:
        print(f"Invalid sample selection: {exc}", file=sys.stderr)
        return 2
    if not samples:
        print("No samples selected.", file=sys.stderr)
        return 2

    summary: list[dict[str, Any]] = []
    failed = 0
    for sample in samples:
        result = run_sample(sample, artifact_root)
        summary.append(result)
        print(f"[{result['status'].upper()}] {result['id']}")
        for error in result["errors"]:
            print(f"  - {error}")
        if result["status"] != "passed":
            failed += 1

    if artifact_root is not None:
        summary_path = artifact_root / "summary.json"
        summary_path.write_text(
            json.dumps({"manifest": args.manifest, "results": summary}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"Summary: {len(samples) - failed} passed, {failed} failed, {len(samples)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
