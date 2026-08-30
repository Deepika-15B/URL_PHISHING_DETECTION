"""
scratch/audit_codebase_stages_6_and_7.py
=========================================
Performs Stage 6 and Stage 7 code audits across the repository.
"""
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_TERMS = [
    "title_domain_similarity_score",
    "title_matches_domain",
    "URLSimilarityIndex",
    "DomainTitleMatchScore",
]

DICT_MUTATIONS = [
    ".update",
    ".pop",
    ".setdefault",
    "defaultdict",
    "copy(",
    "deepcopy(",
]


def audit_stage_6():
    results = []
    py_json_files = list(_PROJECT_ROOT.rglob("*.py")) + list(_PROJECT_ROOT.rglob("*.json"))

    for filepath in py_json_files:
        if ".venv" in filepath.parts or "__pycache__" in filepath.parts:
            continue
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            for line_idx, line in enumerate(lines, start=1):
                for term in TARGET_TERMS:
                    if term in line:
                        rel_path = filepath.relative_to(_PROJECT_ROOT)
                        results.append({
                            "term": term,
                            "file": str(rel_path),
                            "abs_path": str(filepath.resolve()),
                            "line_num": line_idx,
                            "code": line.strip()
                        })
        except Exception:
            pass

    return results


def audit_stage_7():
    results = []
    py_files = list(_PROJECT_ROOT.rglob("*.py"))

    for filepath in py_files:
        if ".venv" in filepath.parts or "__pycache__" in filepath.parts:
            continue
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            for line_idx, line in enumerate(lines, start=1):
                for op in DICT_MUTATIONS:
                    if op in line:
                        rel_path = filepath.relative_to(_PROJECT_ROOT)
                        results.append({
                            "op": op,
                            "file": str(rel_path),
                            "abs_path": str(filepath.resolve()),
                            "line_num": line_idx,
                            "code": line.strip()
                        })
        except Exception:
            pass

    return results


def main():
    print("Executing Stage 6 Code Audit...")
    s6 = audit_stage_6()
    print(f"Stage 6 found {len(s6)} occurrences.")

    print("Executing Stage 7 Code Audit...")
    s7 = audit_stage_7()
    print(f"Stage 7 found {len(s7)} occurrences.")

    # Save to JSON for report generation
    import json
    audit_data = {"stage_6": s6, "stage_7": s7}
    out_path = _PROJECT_ROOT / "scratch" / "stages_6_7_audit_results.json"
    out_path.write_text(json.dumps(audit_data, indent=2), encoding="utf-8")
    print(f"Audit results written to {out_path}")


if __name__ == "__main__":
    main()
