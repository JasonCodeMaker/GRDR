"""Lightweight JS and CSV fact helpers for research packages."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RESULT_COLUMNS = [
    "row_id",
    "exp_id",
    "metric",
    "value",
    "unit",
    "split",
    "baseline",
    "verdict",
    "validity",
    "source_artifact",
    "source_mtime",
    "extractor",
    "extracted_at",
]

LIVE_CHECK_COLUMNS = [
    "row_id",
    "time",
    "exp_id",
    "run_id",
    "agent",
    "run_state",
    "last_log",
    "progress",
    "metrics",
    "resource",
    "artifacts",
    "eta",
    "action",
    "next_check",
    "source_artifact",
    "source_mtime",
    "extractor",
    "extracted_at",
]

RESOURCE_ALLOCATION_COLUMNS = [
    "row_id",
    "exp_id",
    "purpose",
    "dependency",
    "target",
    "capacity",
    "assigned",
    "reason",
    "agent",
    "command_cwd_env",
    "session_job",
    "runtime_root",
    "log_path",
    "expected_duration",
    "status",
    "source_artifact",
    "source_mtime",
    "extractor",
    "extracted_at",
]

METHODS_TRIED_COLUMNS = [
    "row_id",
    "exp_id",
    "method",
    "hypothesis",
    "gate",
    "measured",
    "verdict",
    "evidencePath",
    "source_table",
    "source_row",
    "source_artifact",
    "extracted_at",
]

VALID_RESULT_VALIDITY = {"VALID", "PARTIAL", "RESULT_FAIL", "UNMEASURED", "DIAGNOSTIC_ONLY", "MISSING"}
VALID_EXPERIMENT_VERDICT = {"PASS", "FAIL", "INCONCLUSIVE", "DIAGNOSTIC", ""}
VALID_RUN_STATES = {"QUEUED", "RUNNING", "COMPLETED", "RUN_FAILED", "RUN_HALTED", "STALE", "SKIPPED"}


class FactError(RuntimeError):
    """Raised when package fact data is malformed."""


@dataclass(frozen=True)
class FactPaths:
    root: Path
    pkg: str
    package_data_dir: Path
    facts_js: Path
    tables_dir: Path
    extractors_dir: Path


def fact_paths(pkg: str, root: Path | str = Path(".")) -> FactPaths:
    root = Path(root)
    base = root / "research_html" / "data" / "packages"
    package_data_dir = base / pkg
    return FactPaths(
        root=root,
        pkg=pkg,
        package_data_dir=package_data_dir,
        facts_js=base / f"{pkg}.facts.js",
        tables_dir=package_data_dir / "tables",
        extractors_dir=package_data_dir / "extractors",
    )


def table_csv_path(pkg: str, table_name: str, root: Path | str = Path(".")) -> Path:
    return fact_paths(pkg, root=root).tables_dir / f"{table_name}.csv"


def is_fact_backed(pkg: str, root: Path | str = Path(".")) -> bool:
    return fact_paths(pkg, root=root).package_data_dir.exists()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def facts_js_text(pkg: str, facts: dict) -> str:
    payload = json.dumps(facts, indent=2, sort_keys=True, ensure_ascii=False)
    return (
        "window.PACKAGE_FACTS = window.PACKAGE_FACTS || {};\n"
        f"window.PACKAGE_FACTS[{json.dumps(pkg)}] = {payload};\n"
    )


def write_facts_js(pkg: str, facts: dict, root: Path | str = Path(".")) -> Path:
    path = fact_paths(pkg, root=root).facts_js
    atomic_write(path, facts_js_text(pkg, facts))
    return path


def load_facts_js(pkg: str, root: Path | str = Path(".")) -> dict:
    path = fact_paths(pkg, root=root).facts_js
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    pat = re.compile(
        r"window\.PACKAGE_FACTS\[" + re.escape(json.dumps(pkg)) + r"\]\s*=\s*(\{.*\});\s*$",
        re.DOTALL,
    )
    match = pat.search(text)
    if not match:
        raise FactError(f"cannot parse package facts JS: {path}")
    return json.loads(match.group(1))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_csv_fact_row(columns: list[str], row: dict[str, str]) -> None:
    if "validity" in columns:
        validity = str(row.get("validity", "")).strip()
        if validity and validity not in VALID_RESULT_VALIDITY:
            raise FactError(
                f"invalid validity {validity!r}; expected one of {sorted(VALID_RESULT_VALIDITY)}"
            )
    if "verdict" in columns:
        verdict = str(row.get("verdict", "")).strip()
        if verdict and verdict not in VALID_EXPERIMENT_VERDICT:
            raise FactError(
                f"invalid verdict {verdict!r}; expected one of {sorted(VALID_EXPERIMENT_VERDICT)}"
            )


def _normalize_row(columns: list[str], row: dict) -> dict[str, str]:
    row_id = str(row.get("row_id", "")).strip()
    if not row_id:
        raise FactError("CSV fact row requires non-empty row_id")
    normalized = {col: str(row.get(col, "")) for col in columns}
    normalized["row_id"] = row_id
    validate_csv_fact_row(columns, normalized)
    return normalized


def upsert_csv_rows(path: Path, columns: list[str], rows: Iterable[dict]) -> Path:
    if "row_id" not in columns:
        raise FactError("columns must include row_id")
    existing = read_csv_rows(path)
    by_id = {str(row.get("row_id", "")): row for row in existing if row.get("row_id")}
    order = [str(row.get("row_id")) for row in existing if row.get("row_id")]
    for raw in rows:
        row = _normalize_row(columns, raw)
        if row["row_id"] not in by_id:
            order.append(row["row_id"])
        by_id[row["row_id"]] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row_id in order:
            writer.writerow({col: by_id[row_id].get(col, "") for col in columns})
    os.replace(tmp, path)
    return path


def file_revision(path: Path) -> str:
    data = path.read_bytes()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def source_ref(table_id: str, row_id: str) -> str:
    if not table_id or not row_id:
        raise FactError("source_ref requires table_id and row_id")
    return f"{table_id}:{row_id}"


def split_source_ref(ref: str) -> tuple[str, str]:
    table_id, sep, row_id = ref.partition(":")
    if not sep or not table_id or not row_id:
        raise FactError(f"invalid source row ref: {ref!r}")
    return table_id, row_id


def find_row_by_ref(tables_dir: Path, ref: str) -> dict[str, str]:
    table_id, row_id = split_source_ref(ref)
    table_path = tables_dir / f"{table_id}.csv"
    for row in read_csv_rows(table_path):
        if row.get("row_id") == row_id:
            return row
    raise FactError(f"source row not found: {ref}")


def _source_path(pkg: str, source: str, root: Path | str = Path(".")) -> Path:
    root = Path(root)
    paths = fact_paths(pkg, root=root)
    if source.startswith("tables/"):
        return paths.package_data_dir / source
    return root / source


def relative_source_revision(pkg: str, source: str, root: Path | str = Path(".")) -> str:
    path = _source_path(pkg, source, root=root)
    if not path.exists():
        raise FactError(f"projection source missing: {source}")
    return file_revision(path)


def record_page_projection(
    pkg: str,
    page: str,
    sources: Iterable[str],
    html_path: Path,
    renderer: str,
    root: Path | str = Path("."),
) -> Path:
    root = Path(root)
    html_path = Path(html_path)
    if not html_path.exists():
        raise FactError(f"projection HTML missing: {html_path}")
    facts = load_facts_js(pkg, root=root)
    pages = facts.setdefault("projections", {}).setdefault("pages", {})
    pages[page] = {
        "renderer": renderer,
        "sources": {source: relative_source_revision(pkg, source, root=root) for source in sources},
        "htmlRevision": file_revision(html_path),
        "renderedAt": dt.datetime.now(dt.UTC).astimezone().isoformat(timespec="seconds"),
    }
    return write_facts_js(pkg, facts, root=root)


def page_projection(pkg: str, page: str, root: Path | str = Path(".")) -> dict:
    facts = load_facts_js(pkg, root=root)
    return ((facts.get("projections") or {}).get("pages") or {}).get(page) or {}


def assert_page_projection_fresh(pkg: str, page: str, root: Path | str = Path(".")) -> None:
    root = Path(root)
    projection = page_projection(pkg, page, root=root)
    if not projection:
        raise FactError(f"projection metadata missing: {page}")
    for source, expected in (projection.get("sources") or {}).items():
        actual = relative_source_revision(pkg, source, root=root)
        if actual != expected:
            raise FactError(f"stale source for {page}: {source}")
    html_path = root / "research_html" / "packages" / pkg / page
    if not html_path.exists():
        raise FactError(f"projection HTML missing: {page}")
    if file_revision(html_path) != projection.get("htmlRevision"):
        raise FactError(f"stale html for {page}: {page}")
