"""
A clean, human-readable activity log.

CrewAI's own verbose output is a firehose. This subscribes to CrewAI's event bus
and prints one tidy line per step, including WHAT each tool call is doing:

    ▶ Agent 1: Distributor is starting work...
        -> hands to Software Engineer: implement the update_rfid action
    ▶ Agent 2: Software Engineer is starting work...
        editing charge_points_controller.rb
            - def update_rfid; end
            + def update_rfid; charge_point.update(...); end
        $ runs: bundle exec rspec
    ✔ Software Engineer completed its work.
    ...
    ✔ Distributor completed its work.

    ==> WORK IS DONE.

Call enable_agent_tracking() once at startup.
"""
import json

from crewai.events import (AgentExecutionCompletedEvent,
                           AgentExecutionStartedEvent,
                           ToolUsageFinishedEvent, ToolUsageStartedEvent,
                           crewai_event_bus)

_state = {"registered": False, "count": 0}


def _role(event) -> str:
    role = getattr(event, "agent_role", None)
    if role:
        return role
    agent = getattr(event, "agent", None)
    return getattr(agent, "role", None) or "an agent"


def _args(event) -> dict:
    """The tool's arguments as a dict, whatever shape CrewAI passed them in."""
    a = getattr(event, "tool_args", None)
    if isinstance(a, dict):
        return a
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {"_raw": a}
    return {}


def _short(text, n=60):
    return " ".join(str(text).split())[:n]


def _tool_line(role: str, tool: str, args: dict) -> str:
    """A readable description of one tool call - the file + change, the command,
    or the delegation reason. CrewAI passes tool_name as the snake_case function
    name (e.g. 'apply_edit'), so we normalize to that."""
    t = (tool or "").lower().replace(" ", "_")

    # Delegation: show WHO gets the work and WHY (the task text is the reason).
    if "delegate" in t or "ask_question" in t:
        co = args.get("coworker") or args.get("agent") or "a coworker"
        why = _short(args.get("task") or args.get("question") or "", 90)
        return f"    -> hands to {co}: {why}"

    # Editing a file: show the file name and the change.
    if t == "apply_edit":
        path = args.get("path", "?")
        before = _short(args.get("search", ""), 50)
        after = _short(args.get("replace", ""), 50)
        return (f"    editing {path}\n"
                f"        - {before}\n"
                f"        + {after}")
    if t == "create_file":
        return f"    creating {args.get('path', '?')}"

    # Running things: show the exact command.
    if t == "run_command":
        return f"    $ runs: {args.get('command', '')}"
    if t == "run_python_file":
        extra = args.get("args", "")
        return f"    $ runs: python3 {args.get('path','')}{(' ' + extra) if extra else ''}"
    if t == "run_tests":
        return f"    $ runs tests: pytest {args.get('path','')}".rstrip()
    if t == "install_package":
        return f"    installing package(s): {args.get('packages','')}"
    if t == "run_project":
        return "    running the project"

    # Lookups.
    if "google" in t:
        return f"    searching the web: {_short(args.get('query',''), 70)}"
    if t == "read_file":
        return f"    reading {args.get('path','')}"
    if t in ("search_code", "list_project_files"):
        q = args.get("query") or args.get("pattern") or ""
        return f"    {t.replace('_', ' ')}{(': ' + q) if q else ''}"
    if t == "check_changes":
        return "    syntax-checking the changed files"

    return f"    used tool: {tool}"


def _outcome_line(tool: str, output) -> str | None:
    """Show the RESULT of an action tool (so a failed edit is not mistaken for a
    successful one). Read-only tools stay quiet - their start line was enough."""
    t = (tool or "").lower().replace(" ", "_")
    out = str(output or "").strip()
    first = out.splitlines()[0] if out else ""
    if t in ("apply_edit", "create_file"):
        mark = "✓" if first.startswith("OK") else "✗"   # ✓ / ✗
        return f"        {mark} {first[:90]}"
    if t in ("run_command", "run_python_file", "run_tests", "run_project"):
        code = next((l for l in out.splitlines() if l.startswith("exit_code")), "")
        return f"        -> {code}" if code else None
    if t == "install_package":
        return f"        -> {first[:90]}"
    return None


def enable_agent_tracking() -> None:
    """Register the event listeners once per process."""
    if _state["registered"]:
        _state["count"] = 0
        return
    _state["registered"] = True

    @crewai_event_bus.on(AgentExecutionStartedEvent)
    def _on_start(source, event):
        _state["count"] += 1
        print(f"\n▶ Agent {_state['count']}: {_role(event)} is starting work...")

    @crewai_event_bus.on(ToolUsageStartedEvent)
    def _on_tool(source, event):
        print(_tool_line(_role(event), event.tool_name, _args(event)))

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def _on_tool_done(source, event):
        line = _outcome_line(event.tool_name, getattr(event, "output", ""))
        if line:
            print(line)

    @crewai_event_bus.on(AgentExecutionCompletedEvent)
    def _on_done(source, event):
        role = _role(event)
        print(f"✔ {role} completed its work.")
        # The Distributor (manager) finishes last - that means the whole job is done.
        if role == "Distributor":
            print("\n==> WORK IS DONE.")
