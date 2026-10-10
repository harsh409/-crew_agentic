"""
A small, safe file layer for the target project.

This is the CrewAI version's equivalent of the original project's Workspace: it
is the ONLY thing that touches the target repo, and every path is jailed inside
the project root so an agent can never read or write outside it. Kept compact on
purpose - it is meant to be read in one sitting.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

# Directories that are never worth searching or editing.
IGNORE_DIRS = {".git", "node_modules", "vendor", "tmp", "log", "__pycache__",
               ".venv", "venv", "dist", "build", ".idea", ".vscode"}
# Extensions we treat as source.
SOURCE_SUFFIXES = {".rb", ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".java",
                   ".rake", ".yml", ".yaml", ".json", ".erb"}
# Data files: agents must be able to SEE these (so they know the data exists),
# even though they are not code. Listed by name; their (often huge) contents are
# not slurped during search.
DATA_SUFFIXES = {".csv", ".tsv", ".parquet", ".jsonl", ".txt", ".xlsx", ".feather"}
# Files larger than this are listed but their contents are not read during search.
MAX_SEARCH_BYTES = 2_000_000


# How to syntax-check a file, per extension. {path} is filled in at run time.
# A language whose toolchain is missing is simply skipped (treated as ok).
SYNTAX_CHECKS = {
    ".py": ("python", ["python3", "-m", "py_compile", "{path}"]),
    ".rb": ("ruby", ["ruby", "-c", "{path}"]),
    ".js": ("node", ["node", "--check", "{path}"]),
    ".jsx": ("node", ["node", "--check", "{path}"]),
    ".ts": ("node", ["node", "--check", "{path}"]),
}


class Workspace:
    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Not a directory: {self.root}")
        # Files this run has edited/created, so the Tester knows what to check.
        self.edited: List[str] = []

    # ---- paths -------------------------------------------------------------

    def resolve(self, rel: str) -> Path:
        """Turn a repo-relative path into an absolute one, jailed to the root."""
        rel = rel.strip().strip("`").lstrip("/")
        path = (self.root / rel).resolve()
        if self.root not in path.parents and path != self.root:
            raise ValueError(f"Path escapes the project: {rel}")
        return path

    # ---- reading -----------------------------------------------------------

    def list_files(self) -> List[str]:
        out: List[str] = []
        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(self.root)
            # Only ignore dirs that sit INSIDE the project - not parts of the
            # absolute path above the root (e.g. a project under /tmp).
            if any(part in IGNORE_DIRS for part in rel.parts[:-1]):
                continue
            if p.suffix.lower() in SOURCE_SUFFIXES or p.suffix.lower() in DATA_SUFFIXES:
                out.append(str(rel))
        return out

    def read(self, rel: str) -> str:
        try:
            return self.resolve(rel).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"(could not read {rel}: {exc})"

    def search(self, query: str, max_results: int = 8) -> List[Tuple[str, float]]:
        """Cheap keyword ranking: score files by how many query words they hit
        (in the path and the text). No LLM, no embeddings - just enough to point
        the agents at the right files."""
        words = [w.lower() for w in query.replace("/", " ").replace("_", " ").split()
                 if len(w) >= 3]
        scored: List[Tuple[str, float]] = []
        for rel in self.list_files():
            score = 0.0
            low = rel.lower()
            for w in words:
                if w in low:
                    score += 5           # a path hit is worth a lot
            try:
                path = self.resolve(rel)
                # Do not slurp giant data files (e.g. multi-MB CSVs) into memory
                # just to keyword-score them; the path hit above still ranks them.
                if path.stat().st_size <= MAX_SEARCH_BYTES:
                    text = path.read_text(encoding="utf-8", errors="ignore").lower()
                else:
                    text = ""
            except Exception:
                text = ""
            for w in words:
                score += text.count(w)
            if score:
                scored.append((rel, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:max_results]

    # ---- writing (with a backup, always) -----------------------------------

    def _backup(self, rel: str) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dest = Path.home() / ".crew_agent_backups" / self.root.name / stamp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = self.resolve(rel)
        if src.is_file():
            shutil.copy2(src, dest)
        return str(dest)

    def apply_edit(self, rel: str, search: str, replace: str) -> str:
        """Replace a snippet in a file (git-style search/replace).

        Match is whitespace-tolerant: an exact match is tried first, then a
        match that ignores differences in indentation and run-length of
        whitespace/newlines. Without this, a SEARCH snippet that is off by one
        space silently fails - which is why edits could look done but never
        actually write."""
        path = self.resolve(rel)
        if not path.is_file():
            return f"ERROR: {rel} does not exist."
        text = path.read_text(encoding="utf-8")
        new_text = self._replace_tolerant(text, search, replace)
        if new_text is None:
            return (f"ERROR: the SEARCH text was not found in {rel} (even "
                    f"ignoring whitespace). Read the file again and copy a real "
                    f"snippet from it.")
        if new_text == text:
            return f"OK: {rel} already matches the desired change - nothing to do."
        backup = self._backup(rel)
        path.write_text(new_text, encoding="utf-8")
        if rel not in self.edited:
            self.edited.append(rel)
        return f"OK: edited {rel} (backup at {backup})"

    @staticmethod
    def _replace_tolerant(text: str, search: str, replace: str):
        """Replace the first occurrence of `search` in `text`, tolerating
        whitespace/indentation differences. Returns the new text, or None if no
        match is found."""
        if search in text:                      # 1. exact
            return text.replace(search, replace, 1)
        # 2. whitespace-flexible: match the non-whitespace tokens of `search`
        # with any whitespace (incl. newlines/indentation) between them.
        tokens = search.split()
        if not tokens:
            return None
        pattern = r"\s+".join(re.escape(tok) for tok in tokens)
        m = re.search(pattern, text)
        if not m:
            return None
        return text[:m.start()] + replace + text[m.end():]

    def create_file(self, rel: str, content: str) -> str:
        path = self.resolve(rel)
        if path.exists():
            # Idempotent: re-creating an identical file (a retry / model re-route)
            # is a no-op success, not an error - so the agent moves on instead of
            # trying to "fix" it. Only a genuine content change is refused.
            if path.read_text(encoding="utf-8", errors="replace") == content:
                return f"OK: {rel} already exists with the same content - nothing to do."
            return (f"ERROR: {rel} already exists with different content; "
                    f"use apply_edit to change it.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        if rel not in self.edited:
            self.edited.append(rel)
        return f"OK: created {rel}"

    # ---- verifying ---------------------------------------------------------

    def check_syntax(self, rel: str) -> dict:
        """Syntax-check one file with its language's compiler. No LLM."""
        suffix = Path(rel).suffix.lower()
        spec = SYNTAX_CHECKS.get(suffix)
        if not spec:
            return {"path": rel, "ok": True, "checked": False,
                    "detail": f"no checker for {suffix or 'this type'}"}
        lang, template = spec
        argv = [a.replace("{path}", str(self.resolve(rel))) for a in template]
        code, out = self.run(argv, timeout=60)
        if code == -1 and ("No such file" in out or "not found" in out):
            return {"path": rel, "ok": True, "checked": False,
                    "detail": f"{lang} toolchain not installed"}
        return {"path": rel, "ok": code == 0, "checked": True,
                "detail": "" if code == 0 else out[:800]}

    def check_edited(self) -> dict:
        """Syntax-check every file edited this run. The Tester's hard gate."""
        results = [self.check_syntax(rel) for rel in self.edited]
        failures = [r for r in results if r["checked"] and not r["ok"]]
        return {"results": results, "ok": not failures,
                "failures": failures, "edited": list(self.edited)}

    # ---- running -----------------------------------------------------------

    def run(self, argv: List[str], timeout: int = 60) -> Tuple[int, str]:
        try:
            proc = subprocess.run(argv, cwd=str(self.root), capture_output=True,
                                  text=True, timeout=timeout,
                                  stdin=subprocess.DEVNULL)
            return proc.returncode, (proc.stdout + proc.stderr)[-4000:]
        except subprocess.TimeoutExpired:
            return -1, f"timed out after {timeout}s"
        except FileNotFoundError as exc:
            return -1, str(exc)
