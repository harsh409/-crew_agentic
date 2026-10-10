"""
CrewAI tools - the hands the agents use to touch the real repo.

An LLM can only produce text; a *tool* is how CrewAI lets an agent take an
action. Each function below is decorated with @tool, so an agent can call it by
name and get a real result (files listed, code read, an edit applied, the
project run). They all go through one Workspace, so every path stays jailed.

The Workspace is set once at startup by `set_workspace()` and shared by all the
tools (a module-level singleton keeps the tool signatures simple).
"""
import re
import shlex
from pathlib import Path

from crewai.tools import tool

from workspace import Workspace

_WS: Workspace | None = None
# Per-run cache for web searches: when a model retries or the gateway routes to
# another model, agents often re-issue the SAME query. Serving it from cache
# makes those repeats instant (and signals to the agent it already ran this).
_SEARCH_CACHE: dict = {}


def set_workspace(ws: Workspace) -> None:
    global _WS
    _WS = ws
    _SEARCH_CACHE.clear()


def _ws() -> Workspace:
    if _WS is None:
        raise RuntimeError("Workspace not set - call set_workspace() first.")
    return _WS


def get_workspace() -> Workspace:
    return _ws()


@tool("List project files")
def list_project_files(pattern: str = "") -> str:
    """List the source files in the project (relative paths). Optionally pass a
    substring `pattern` to filter (e.g. 'controller'). Use this to see what
    exists before searching or editing.
    """
    files = _ws().list_files()
    if pattern:
        files = [f for f in files if pattern.lower() in f.lower()]
    return "\n".join(files[:400]) or "(no matching source files)"


@tool("Search code")
def search_code(query: str) -> str:
    """Find the files most relevant to a query (keywords, a class name, an error
    message). Returns ranked 'path (score)' lines. Use it to locate the code a
    task is about.
    """
    hits = _ws().search(query)
    if not hits:
        return "(no matches)"
    return "\n".join(f"{rel} (score {score:g})" for rel, score in hits)


# Token thrift: a runaway file should not blow the context budget. 20k chars is
# plenty for the Coder to make an exact edit; anything larger is truncated.
MAX_READ_CHARS = 20000


@tool("Read file")
def read_file(path: str) -> str:
    """Read a file's full contents by its repo-relative path. Always read a file
    before editing it, so your SEARCH snippet is exact.
    """
    text = _ws().read(path)
    if len(text) > MAX_READ_CHARS:
        return text[:MAX_READ_CHARS] + f"\n... (truncated at {MAX_READ_CHARS} chars)"
    return text


@tool("Apply edit")
def apply_edit(path: str, search: str, replace: str) -> str:
    """Edit a file by replacing an EXACT snippet. `search` must be copied
    verbatim from the current file (including indentation); it is replaced by
    `replace`. The original is backed up first. Returns OK or an error to fix.
    """
    return _ws().apply_edit(path, search, replace)


@tool("Create file")
def create_file(path: str, content: str) -> str:
    """Create a NEW file with the given content. Fails if the file already
    exists (edit it instead).
    """
    return _ws().create_file(path, content)


@tool("Check changes")
def check_changes(path: str = "") -> str:
    """Syntax-check the files changed this run, using the real language compiler
    (no LLM). Pass a `path` to check just that file, or leave it empty to check
    everything changed. Returns PASS or, for each broken file, the compiler
    error. Call this after edits; only proceed if it PASSes.
    """
    ws = _ws()
    if path:
        r = ws.check_syntax(path)
        if not r["checked"]:
            return f"{path}: skipped ({r['detail']})"
        return f"{'PASS' if r['ok'] else 'FAIL'}\n- {path}: " + (
            "OK" if r["ok"] else "FAILED\n" + r["detail"])
    report = ws.check_edited()
    if not report["edited"]:
        return "No files have been changed yet."
    lines = []
    for r in report["results"]:
        if not r["checked"]:
            lines.append(f"- {r['path']}: skipped ({r['detail']})")
        elif r["ok"]:
            lines.append(f"- {r['path']}: OK")
        else:
            lines.append(f"- {r['path']}: FAILED\n{r['detail']}")
    verdict = "PASS" if report["ok"] else "FAIL"
    return f"{verdict}\n" + "\n".join(lines)


def _search_duckduckgo(query: str, n: int) -> list:
    """(title, url, snippet) tuples from DuckDuckGo. Reliable, no API key.
    """
    from ddgs import DDGS
    with DDGS() as d:
        hits = d.text(query, max_results=n)
    return [(h.get("title", ""), h.get("href", ""), h.get("body", "")[:200])
            for h in hits]


def _search_google(query: str, n: int) -> list:
    """Fallback: scrape Google via googlesearch-python (often rate-limited)."""
    from googlesearch import search
    return [(getattr(r, "title", ""), getattr(r, "url", ""),
             (getattr(r, "description", "") or "")[:200])
            for r in search(query, num_results=n, advanced=True, timeout=8)]


@tool("Google search")
def google_search(query: str, num_results: int = 5) -> str:
    """Search the web for a query and return the top results (title, URL, and a
    short snippet). Use it when you need external information to implement
    something correctly - a library's API, an install command, an error's
    meaning. Uses DuckDuckGo (reliable), falling back to Google scraping.
    """
    n = max(1, min(num_results, 8))
    key = (query.strip().lower(), n)
    if key in _SEARCH_CACHE:
        return (_SEARCH_CACHE[key]
                + "\n(already searched this in the current run - reuse these "
                  "results; do not search for it again.)")
    results, errors = [], []
    for name, fn in (("DuckDuckGo", _search_duckduckgo), ("Google", _search_google)):
        try:
            results = fn(query, n)
            if results:
                break
        except ImportError as exc:
            errors.append(f"{name}: not installed ({exc})")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if not results:
        return (f"web search returned nothing for {query!r}. "
                + (" / ".join(errors) if errors else "Try a simpler query."))
    out = "\n".join(f"- {title}\n  {url}\n  {snippet}"
                    for title, url, snippet in results)
    _SEARCH_CACHE[key] = out
    return out


# Only these characters are allowed in a package spec, so nothing can be smuggled
# into the pip command (no shell metacharacters, spaces handled by splitting).
_PKG_RE = re.compile(r"^[A-Za-z0-9_.\- \[\]<>=!~]+$")


@tool("Install package")
def install_package(packages: str) -> str:
    """Install one or more Python packages with pip (space- or comma-separated,
    e.g. 'fpdf pytest'). Use this yourself when a run or test fails with
    ModuleNotFoundError - install the missing package, then try again. Returns
    pip's exit code and output.
    """
    names = [p for p in re.split(r"[\s,]+", packages.strip()) if p]
    safe = [n for n in names if _PKG_RE.match(n)]
    if not safe:
        return f"No valid package name in {packages!r}."
    code, out = _ws().run(["python3", "-m", "pip", "install", *safe], timeout=300)
    status = "installed" if code == 0 else "FAILED"
    return f"{status} ({' '.join(safe)}); exit_code={code}\n{out[-1500:]}"


@tool("Run tests")
def run_tests(path: str = "") -> str:
    """Run the project's Python tests with pytest (optionally limited to `path`.
    Returns the exit code and output. If it fails with ModuleNotFoundError,
    install the missing package with install_package and run this again.
    """
    argv = ["python3", "-m", "pytest", "-q"] + ([path] if path else [])
    code, out = _ws().run(argv, timeout=300)
    return f"exit_code={code}\n{out[-2500:]}"


# Commands that can destroy the user's files/data. Blocked outright - an agent
# must never delete, wipe, or overwrite existing files while doing a task.
_DESTRUCTIVE_PROGS = {
    "rm", "rmdir", "unlink", "shred", "srm", "dd", "mkfs", "truncate", "fdisk",
    "mkfs.ext4", "wipefs", "chown", "chmod",
}
_DESTRUCTIVE_RE = re.compile(
    r"\brm\b|\brmdir\b|\bshred\b|\bunlink\b|\bmkfs|\bdd\b|\btruncate\b|"
    r"\bgit\s+clean\b|\bfind\b.*-delete|\b-delete\b|>\s*\S|\bmv\b", re.I)


def _is_destructive(argv: list) -> bool:
    import os as _os
    prog = _os.path.basename(argv[0]) if argv else ""
    if prog in _DESTRUCTIVE_PROGS:
        return True
    if prog == "find" and "-delete" in argv:
        return True
    if prog == "git" and "clean" in argv:
        return True
    # Shell wrappers that could smuggle a destructive command back in.
    if prog in ("sh", "bash", "zsh", "fish") and any(
            _DESTRUCTIVE_RE.search(a) for a in argv[1:]):
        return True
    return False


@tool("Run command")
def run_command(command: str) -> str:
    """Run ANY non-destructive command in the project directory - not just Python.
    Examples: 'pytest -q', 'npm install', 'node index.js', 'go build ./...',
    'make', './my_binary --help'. Runs with the project as its working directory;
    returns the exit code and combined output. (Parsed like a shell line; pipes
    and redirects are not supported - run one command.)

    Destructive commands (rm, rmdir, shred, dd, mkfs, git clean, find -delete,
    etc.) are REFUSED - never delete, wipe, or overwrite the user's files/data."""
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return f"ERROR: could not parse command: {exc}"
    if not argv:
        return "ERROR: empty command."
    if _is_destructive(argv):
        return (f"REFUSED: '{command}' is a destructive command and is not allowed. "
                f"Never delete, move, or overwrite existing files or data. If a file "
                f"needs replacing, use apply_edit/create_file, or leave it and report.")
    code, out = _ws().run(argv, timeout=300)
    return f"$ {command}\nexit_code={code}\n{out}"


@tool("Run python file")
def run_python_file(path: str, args: str = "") -> str:
    """Run a SPECIFIC Python file by its repo-relative path, optionally passing
    `args` to it. Use this when asked to run a particular file, e.g.
    run_python_file('scanner.py', 'AAPL') or ('people_counter/main.py', 'img.jpg').
    Returns the exit code and output.
    """
    # Guard against common misuse (agents trying to run inline code / flags here).
    if path.strip().startswith("-") or path.strip() in ("python3", "python"):
        return ("ERROR: run_python_file runs a .py FILE, not flags. To run inline "
                "Python use run_command with: python3 -c \"<code>\".")
    if any(tok in args for tok in ("import ", "print(", ";", "open(", "exec(", "=")):
        return ("ERROR: `args` are command-line arguments for the script, not Python "
                "code. To execute inline code use run_command: python3 -c \"<code>\".")
    ws = _ws()
    try:
        target = ws.resolve(path)
    except Exception:
        return (f"ERROR: {path} is outside the project. run_python_file only runs a "
                f".py file inside the project. To run an arbitrary command use run_command.")
    if not target.is_file():
        return f"ERROR: {path} is not a file in the project."
    if target.suffix != ".py":
        return f"ERROR: {path} is not a .py file."
    rel = target.relative_to(ws.root)
    argv = ["python3", str(rel)] + (args.split() if args else [])
    code, out = ws.run(argv, timeout=120)
    return f"ran `python3 {rel}{(' ' + args) if args else ''}`\nexit_code={code}\n{out}"


@tool("Run project")
def run_project(args: str = "") -> str:
    """Run the project's main entry point to check it works (use run_python_file
    for a specific named file). Optional `args` are passed to it. Tries
    `python3 main.py`, then `app.py`, then `npm start`. Returns exit code and
    output.
    """
    ws = _ws()
    argv = None
    for name in ("main.py", "app.py"):
        if (ws.root / name).is_file():
            argv = ["python3", name]
            break
    if argv is None and (ws.root / "package.json").is_file():
        argv = ["npm", "start", "--silent"]
    if argv is None:
        return "Could not determine how to run this project."
    if args:
        argv += args.split()
    code, out = ws.run(argv)
    return f"exit_code={code}\n{out}"

