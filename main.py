#!/usr/bin/env python3
"""
CrewAI edition - a multi-agent coding assistant on freeLLMAPI.

    # edit-tasks (Scope -> Coder -> Tester -> Reviewer)
    python3 main.py -p ~/Desktop/app "fix the bug in calc.py"

    # questions (answer only, nothing is changed)
    python3 main.py -p ~/Desktop/app "how do I run this project?"
    python3 main.py -p ~/Desktop/app --question "explain what crew.py does"

    # sessions
    python3 main.py --list
    python3 main.py --continue 1 "now also handle the empty input"
    python3 main.py --show 1
    python3 main.py --delete 1

    # interactive (one saved session for the whole chat)
    python3 main.py -i -p ~/Desktop/app

    # long / multi-line prompts (terminal input can't handle these well):
    python3 main.py -p ~/Desktop/app -f prompt.txt        # read the task from a file
    python3 main.py -p ~/Desktop/app - < prompt.txt       # read the task from stdin
    cat prompt.txt | python3 main.py -p ~/Desktop/app     # pipe it in
    # ...and inside interactive mode, type /edit to compose in your $EDITOR.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from crew import run_task
from sessions import SessionStore, Turn
from token_saver import build_history
from tracker import enable_agent_tracking


def compose_in_editor(initial: str = "") -> str:
    """Open the user's $EDITOR to write a long/multi-line prompt, and return it.
    Falls back to nano. This is how you enter a big prompt comfortably instead of
    fighting the one-line terminal input."""
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
        f.write(initial)
        path = f.name
    try:
        subprocess.call([editor, path])
        text = Path(path).read_text()
    except Exception as exc:
        print(f"Could not open editor {editor!r}: {exc}")
        text = ""
    finally:
        os.unlink(path)
    return text.strip()


def resolve_task_text(args):
    """Work out the task text from the friendliest available source: --file,
    stdin ('-' or a pipe), or the positional argument."""
    if args.file:
        return Path(args.file).expanduser().read_text().strip()
    if args.task == "-":                       # explicit stdin
        return sys.stdin.read().strip()
    if args.task:
        return args.task
    # No task given: if something was piped in (not a terminal), read it.
    if not args.interactive and not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return data
    return None


def _remap_focus(raw_paths, root):
    """Make session file paths resolve under `root` (which may be narrower than
    the root they were saved with). Tries the path as-is, without its first
    folder, then just the basename - keeps whatever actually exists."""
    root = Path(root)
    out = []
    for f in raw_paths:
        for cand in (f, "/".join(f.split("/")[1:]), Path(f).name):
            if cand and (root / cand).exists():
                out.append(cand)
                break
    return sorted(set(out))


def resolve_root(session, explicit_project):
    """The directory a continued turn should be scoped to.

    An explicit -p wins. Otherwise, if every file the session changed lives under
    one top-level sub-folder that exists, scope to THAT folder (so continuing a
    'new project' session doesn't crawl the whole parent directory)."""
    if explicit_project:
        return Path(explicit_project).expanduser()
    root = Path(session.project_dir)
    raw = {f.split("/")[0] for t in session.turns for f in (t.changed_files or [])
           if "/" in f}
    if len(raw) == 1 and (root / next(iter(raw))).is_dir():
        return root / next(iter(raw))
    return root


def _do_turn(task, project_dir, store, session, answer_only=False):
    """Run a single turn. The Distributor decides if it's a question or a task."""
    # Grounding for a continuation: most recent turn(s) in full + an
    # understanding-based summary of the older turns (not a truncation).
    history_brief = build_history(session) if session.turns else ""
    # Files earlier turns touched - a follow-up is almost always about these, so
    # the crew reads them directly instead of crawling the whole project. Remap
    # to the (possibly narrowed) root so the paths resolve.
    raw = sorted({f for t in session.turns for f in (t.changed_files or [])})
    focus_files = _remap_focus(raw, project_dir)

    print("=" * 64)
    print(f"  CrewAI assistant  |  Distributor deciding...")
    print(f"  Project: {project_dir}")
    if focus_files:
        print(f"  Focus: {', '.join(focus_files[:5])}"
              + (" …" if len(focus_files) > 5 else ""))
    print("=" * 64)

    result = run_task(task, str(project_dir), history_brief, answer_only, focus_files)

    print("\n" + "=" * 64)
    print("RESULT")
    print("=" * 64)
    print(result["output"])
    print(f"\nHandled as: {result['intent']}")
    if result["intent"] == "task":
        print(f"Changed files: {', '.join(result['changed_files']) or '(none)'}")
        print(f"Syntax check:  {'PASS' if result['syntax_ok'] else 'FAIL'}")

    session.add_turn(Turn(task=task, intent=result["intent"],
                          output=result["output"][:600],
                          changed_files=result["changed_files"]))
    store.save(session)
    print(f"\nSession: {session.id}  (continue with: --continue {session.id})")
    return result


def print_list(store):
    sessions = store.list()
    if not sessions:
        print("No saved sessions yet.")
        return
    print(f"\nSaved sessions ({len(sessions)}), newest first:\n")
    for i, s in enumerate(sessions, 1):
        print(f"  [{i}] {s.id}")
        print(f"       {s.title}")
        print(f"       {len(s.turns)} turn(s) | updated {s.updated_at}")
        print(f"       project: {s.project_dir}")
    print("\nContinue: python3 main.py --continue <n|id> \"...\"")


def print_show(store, ref):
    s = store.load(ref)
    if not s:
        print(f"No session matching {ref!r}.")
        return
    print(f"\nSession {s.id}\nTitle: {s.title}\nProject: {s.project_dir}")
    for i, t in enumerate(s.turns, 1):
        print("\n" + "-" * 50)
        print(f"Turn {i} ({t.timestamp}) - {t.intent}")
        print(f"  task:    {t.task}")
        if t.changed_files:
            print(f"  changed: {', '.join(t.changed_files)}")
        if t.output:
            print(f"  output:  {t.output[:300]}")


def interactive(store, session, project_dir, answer_only):
    print("\nInteractive mode - type 'quit' to exit. All turns share one session.")
    print("For a long/multi-line prompt: type /edit (opens your editor) or "
          "/file <path>.")
    print(f"Project: {project_dir}")
    while True:
        try:
            task = input("\nTask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if task.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        if task == "/edit":                       # compose a long prompt in $EDITOR
            task = compose_in_editor()
            if task:
                print(f"(got {len(task)} chars from the editor)")
        elif task.startswith("/file "):           # load a prompt from a file
            path = Path(task[6:].strip()).expanduser()
            if not path.is_file():
                print(f"No such file: {path}")
                continue
            task = path.read_text().strip()
            print(f"(loaded {len(task)} chars from {path})")
        if not task:
            continue
        _do_turn(task, project_dir, store, session, answer_only)


def main():
    p = argparse.ArgumentParser(description="CrewAI coding assistant (freeLLMAPI)")
    p.add_argument("task", nargs="?",
                   help="what you want done (task or question); use '-' to read "
                        "it from stdin")
    p.add_argument("--file", "-f", metavar="PATH",
                   help="read the task/prompt from a file (best for long, "
                        "multi-line prompts)")
    p.add_argument("--project", "-p", help="target project directory")
    p.add_argument("--question", action="store_true",
                   help="force answer-only mode (never edits or creates files)")
    p.add_argument("--interactive", "-i", action="store_true",
                   help="chat mode; every turn is saved in one session")
    p.add_argument("--list", "-l", action="store_true", help="list saved sessions")
    p.add_argument("--continue", "-c", dest="continue_id", metavar="ID",
                   help="reopen a session (by number or id) and add a follow-up")
    p.add_argument("--show", metavar="ID", help="show a session's turns")
    p.add_argument("--delete", metavar="ID", help="delete a session")
    args = p.parse_args()
    enable_agent_tracking()   # clean "Agent X is starting / completed" logging

    store = SessionStore()

    # Management commands: act and exit.
    if args.list:
        print_list(store); return
    if args.show:
        print_show(store, args.show); return
    if args.delete:
        s = store.load(args.delete)
        if s and store.delete(s.id):
            print(f"Deleted {s.id}.")
        else:
            print(f"No session matching {args.delete!r}.")
        return

    answer_only = bool(args.question)
    task_text = resolve_task_text(args)

    # Continue an existing session (project comes from it).
    if args.continue_id:
        session = store.load(args.continue_id)
        if not session:
            print(f"No session matching {args.continue_id!r}. Try --list.")
            sys.exit(1)
        project_dir = resolve_root(session, args.project)
        if not project_dir.is_dir():
            print(f"Session's project no longer exists: {project_dir}")
            sys.exit(1)
        print(f"Continuing session {session.id} ({len(session.turns)} turn(s))")
        print(f"Scope: {project_dir}")
        if args.interactive:
            interactive(store, session, project_dir, answer_only)
        elif task_text:
            _do_turn(task_text, project_dir, store, session, answer_only)
        else:
            print("Give a follow-up task, or use -i for interactive.")
        return

    # Fresh run: a project is required.
    if not args.project:
        print("Error: --project/-p is required (or use --continue / --list).")
        sys.exit(1)
    project_dir = Path(args.project).expanduser()
    if not project_dir.is_dir():
        print(f"Error: not a directory: {project_dir}")
        sys.exit(1)

    if args.interactive:
        session = store.create("(interactive)", project_dir)
        interactive(store, session, project_dir, answer_only)
    elif task_text:
        session = store.create(task_text, project_dir)
        _do_turn(task_text, project_dir, store, session, answer_only)
    else:
        print("Nothing to do: give a task, use -f <file>, pipe one in, or -i.")


if __name__ == "__main__":
    main()
