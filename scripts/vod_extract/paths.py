"""Where the main project (dataset + pipeline) lives, in ONE place.

The extractor is a separate repository that reads the main repository
(`mpl-s18-draft-project`) for game metadata, hero vocabulary, `pipeline_lib`
and the audit frames, and writes its output under that repository's
`data/extractions/`. It never modifies anything else there.

Default: a sibling folder named `mpl-s18-draft-project`. Override with the
environment variable MPL_PROJECT_ROOT (PowerShell:
    $env:MPL_PROJECT_ROOT = "D:\\somewhere\\mpl-s18-draft-project"
).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

EXTRACTOR_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    env = os.environ.get("MPL_PROJECT_ROOT")
    root = Path(env) if env else EXTRACTOR_ROOT.parent / "mpl-s18-draft-project"
    if not (root / "scripts" / "pipeline_lib.py").exists():
        raise SystemExit(f"main project not found at {root} (expected scripts/pipeline_lib.py). "
                         "Put it next to this repository or set MPL_PROJECT_ROOT.")
    return root


def ensure_project_scripts_on_path() -> Path:
    root = project_root()
    scripts = str(root / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return root
