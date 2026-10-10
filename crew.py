"""
The Crew - one hierarchical team led by the Distributor.

There is no if/else that decides "question vs task" any more. We hand the whole
request to a manager agent (the Distributor) running a HIERARCHICAL crew, and
CrewAI gives it delegation tools so IT decides who handles the work:

    Distributor (manager)
      ├─ Code Explainer   (questions - read-only)
      ├─ Scope Analyst    ┐
      ├─ Software Engineer│  (code changes, in order)
      ├─ Test Engineer    │
      └─ Code Reviewer    ┘

`history_brief` (condensed by the token saver) grounds a continued session.
`answer_only` is a soft steer from --question; the read-only guarantee still
comes from the Explainer simply having no editing tools.
"""
from crewai import Crew, Process, Task

from agents import build_agents
from tools import get_workspace, set_workspace
from workspace import Workspace


def build_crew(task: str, agents: dict, history_brief: str = "",
               answer_only: bool = False, focus_files=None) -> Crew:
    ground = f"\n\n{history_brief}\n" if history_brief else ""
    steer = ("\nThe user wants an ANSWER ONLY: do not modify or create any file; "
             "delegate to the Code Explainer.\n" if answer_only else "")
    # Files earlier turns of this session created/changed. A follow-up almost
    # always concerns THESE files, so point the crew straight at them instead of
    # letting it crawl the whole project.
    focus = ""
    if focus_files:
        listed = "\n".join(f"  - {f}" for f in focus_files[:25])
        focus = (f"\nRELEVANT FILES (created/changed earlier in this session - "
                 f"this follow-up is almost certainly about them; read THESE "
                 f"directly to answer or change, and do NOT list or search the "
                 f"whole project):\n{listed}\n")

    request = Task(
        description=(
            f"The user's request about the project:\n\"{task}\"\n"
            f"{ground}{focus}{steer}\n"
            "First decide the request's NATURE, then delegate ONLY the "
            "specialists it needs:\n"
            "- QUESTION / explanation: delegate to the Code Explainer (read-only; "
            "nothing is changed) and return the answer. If RELEVANT FILES are "
            "listed above, read those directly - do not crawl the project.\n"
            "- CHANGE to existing code: Scope Analyst -> Software Engineer -> "
            "Test Engineer (check_changes must PASS) -> Code Reviewer.\n"
            "- NEW PROJECT (create a new tool/app): do NOT scope or inspect "
            "existing files. Delegate to the Project Scaffolder, who makes a new "
            "sub-folder and writes the files, then Test Engineer -> Code Reviewer.\n"
            "- MAKE / GENERATE / PRODUCE an artifact with no dedicated tool (a PDF, "
            "a chart, a converted file, scraped data, etc.): delegate to the "
            "Automation Engineer, who writes a Python script, installs any missing "
            "free packages, runs it, and produces the real output.\n"
            "Return the final answer or a summary of the verified change."
        ),
        expected_output="The answer to a question, or a summary of the verified change.",
    )

    # The manager leads; the workers are the crew. (Manager is NOT in this list
    # and has no tools - CrewAI's rules for a hierarchical crew.)
    return Crew(
        agents=[agents["explainer"], agents["scope"], agents["coder"],
                agents["scaffolder"], agents["tester"], agents["command_runner"],
                agents["reviewer"], agents["automation"]],
        tasks=[request],
        process=Process.hierarchical,
        manager_agent=agents["distributor"],
        verbose=False,
    )


def run_task(task: str, project_dir: str, history_brief: str = "",
             answer_only: bool = False, focus_files=None) -> dict:
    """Run one turn. Returns {intent, output, changed_files, syntax_ok}."""
    set_workspace(Workspace(project_dir))
    agents = build_agents()
    crew = build_crew(task, agents, history_brief, answer_only, focus_files)
    result = crew.kickoff()

    # Mechanical hard gate, independent of what the agents claimed: whatever got
    # edited must actually compile. `intent` is inferred from what happened -
    # files changed => it was a task, otherwise it was a question.
    ws = get_workspace()
    check = ws.check_edited()
    return {"intent": "task" if ws.edited else "question",
            "output": str(result), "changed_files": list(ws.edited),
            "syntax_ok": check["ok"], "syntax_report": check}