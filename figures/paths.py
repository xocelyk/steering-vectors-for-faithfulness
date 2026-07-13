"""Shared path resolution for the paper-artifact scripts in figures/.

Generated artifacts are written to figures/out/ (tracked in this repo, so
every figure/table is versioned alongside the code and data that produced
it). figures/generate.sh then syncs figures/out/ into the paper repo's
\\graphicspath directory, which is where main.tex reads them.

  artifacts_dir()       -- where scripts write: figures/out/, unless
                           ARTIFACTS_DIR overrides it (absolute path, or a
                           name resolved against this repo's root -- useful
                           for scratch runs that shouldn't touch anything).

  paper_root()          -- the Overleaf/paper repo (the dir with main.tex).
                           Resolved from, in order:
                             1. PAPER_DIR (environment, or figures/.env)
                             2. the parent of this repo, if it holds main.tex
                                (legacy layout: experiments repo cloned inside
                                the paper repo)
                             3. a unique sibling directory of this repo
                                holding main.tex (current layout: the two
                                repos side by side)

  paper_artifacts_dir() -- where the paper reads artifacts: paper_root() /
                           the directory named in main.tex's \\graphicspath.
                           Bumping the date there is enough to retarget the
                           sync.
"""
import os
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent      # <experiments-repo>/figures/
_REPO = _HERE.parent                         # experiments repo root


def _load_env(path: Path) -> None:
    """Minimal .env loader (stdlib only). Real env vars take precedence."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env(_HERE / ".env")


def artifacts_dir() -> Path:
    """Return (creating if needed) the directory scripts write artifacts to."""
    name = os.environ.get("ARTIFACTS_DIR")
    if name:
        p = Path(name).expanduser()
        out = p if p.is_absolute() else _REPO / p
    else:
        out = _HERE / "out"
    out.mkdir(parents=True, exist_ok=True)
    return out


def paper_root() -> Path:
    """Return the paper (Overleaf) repo root, i.e. the dir holding main.tex."""
    env = os.environ.get("PAPER_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if not (p / "main.tex").exists():
            raise SystemExit(f"PAPER_DIR={p} does not contain main.tex")
        return p
    if (_REPO.parent / "main.tex").exists():
        return _REPO.parent
    siblings = [
        d for d in _REPO.parent.iterdir()
        if d.is_dir() and d != _REPO and (d / "main.tex").exists()
    ]
    if len(siblings) == 1:
        return siblings[0]
    raise SystemExit(
        "Could not locate the paper repo (no unique main.tex next to "
        f"{_REPO}). Set PAPER_DIR in {_HERE / '.env'} to the Overleaf repo "
        "path."
    )


def paper_artifacts_dir() -> Path:
    """Return the artifacts dir main.tex reads (from its \\graphicspath)."""
    root = paper_root()
    m = re.search(r"\\graphicspath\{\{([^}]+?)/?\}\}",
                  (root / "main.tex").read_text())
    return root / (m.group(1) if m else "artifacts")
