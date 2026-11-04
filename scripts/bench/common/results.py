"""Write one benchmark result to `benchmarks/results/`.

The file name is `<UTC timestamp>-<component>-<variant>.json`. A result is validated
against the schema and scanned for personal data first; nothing is written when
either check fails, and an existing file is never overwritten.
"""

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.bench.common.privacy import PrivacyContext, PrivacyError, find_personal_data
from scripts.bench.common.schema import TIMESTAMP_FORMAT, validate_result

RESULTS_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "results"


def result_filename(result: Mapping[str, Any]) -> str:
    moment = datetime.strptime(result["created_at"], TIMESTAMP_FORMAT)
    return f"{moment:%Y%m%dT%H%M%SZ}-{result['component']}-{result['variant']}.json"


def write_result(
    result: Mapping[str, Any],
    directory: Path = RESULTS_DIR,
    *,
    context: PrivacyContext | None = None,
) -> Path:
    validate_result(result)
    findings = find_personal_data(result, context or PrivacyContext.current())
    if findings:
        raise PrivacyError("result holds personal data: " + "; ".join(findings))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / result_filename(result)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(result, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    return path
