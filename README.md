# CrewAI Coding Assistant

A multi-agent coding assistant built on **CrewAI**, running against your local
**freeLLMAPI** gateway. This is the CrewAI rewrite of the hand-built project in
`~/Desktop/agent` — same idea, expressed with CrewAI's `Agent` / `Task` / `Crew`
so it is easier to read and extend. **The original project is untouched.**

## How it maps

| Piece | File | Job |
|---|---|---|
| The LLM every agent uses | `llm.py` | wraps LLMAPI as a CrewAI `LLM` (LiteLLM, `openai/` prefix) |
| Safe file layer | `workspace.py` | the only thing that touches the target repo; every path is jailed |
| The agents' hands | `tools.py` | CrewAI `@tool`s: list / search / read / edit / create / run / check / **google_search** |
| The agents | `agents.py` | Distributor (manager) + Explainer, Scope, Coder, Tester, Reviewer |
| The crew | `crew.py` | one **hierarchical** crew: the Distributor delegates to the right specialists |
| Sessions | `sessions.py` | save / list / continue past chats |
| Token saver | `token_saver.py` | condenses long history so big chats stay cheap |
| CLI entry point | `main.py` | tasks, questions, sessions, interactive |

## Long / multi-line prompts

The terminal's one-line input can't handle a big prompt (blank lines end it,
paste breaks). Use any of these instead:

```bash
# 1. from a file (best for long prompts) - write it once, reuse it
python3 main.py -p ~/Desktop/app -f prompt.txt

# 2. from stdin
python3 main.py -p ~/Desktop/app - < prompt.txt
cat prompt.txt | python3 main.py -p ~/Desktop/app

# 3. inside interactive mode, open your $EDITOR to compose it
python3 main.py -i -p ~/Desktop/app
Task> /edit                 # opens nano/vim; write freely, save, close
Task> /file prompt.txt      # or load a prompt file
```

## Modes and options

| Command | What it does |
|---|---|
| `python3 main.py -p <dir> "<task>"` | run an edit-task (Scope → Coder → Tester → Reviewer) |
| `python3 main.py -p <dir> "<question>"` | a question is auto-detected and answered read-only |
| `python3 main.py -p <dir> --question "<q>"` | force answer-only mode (never edits) |
| `python3 main.py -i -p <dir>` | interactive chat; every turn saved in one session |
| `python3 main.py --list` | list saved sessions, newest first |
| `python3 main.py --continue <n\|id> "<follow-up>"` | reopen a session and add a turn |
| `python3 main.py --show <n\|id>` | show a session's turns |
| `python3 main.py --delete <n\|id>` | delete a session |

- **Runs any command** — a dedicated **Command Runner** agent has a `run_command`
  tool that executes *any* shell command in the project dir (not just Python):
  `pytest -q`, `npm install`, `node index.js`, `go build`, `make`, a binary, etc.
  The Distributor delegates anything that needs executing to it; if a command
  fails for a missing package it installs it and retries.
- **Installs its own dependencies** — the Test Engineer runs the project's tests
  (`run_tests`) and, if anything fails with `ModuleNotFoundError`, installs the
  missing package itself (`install_package`, i.e. `pip install`) and retries -
  it never tells you to run pip. The Coder and Scaffolder can install too.
- **Web search** — the Coder, Scaffolder and Explainer have a `google_search`
  tool for looking up an API, an install command, or an error's meaning while
  implementing. It uses **DuckDuckGo** (reliable, no API key) and falls back to
  Google scraping if DuckDuckGo is unavailable.
- **Questions change nothing** — the Explainer has no editing tools at all.
- **The Tester is a hard gate** — every edited file is syntax-checked with the
  real compiler (`py_compile` / `ruby -c` / `node --check`); the result is
  reported as `Syntax check: PASS/FAIL`.
- **Continuing stays in scope** — on `--continue`, the run is scoped to the
  files the session actually touched: if they all live under one sub-folder
  (e.g. a `new project` created `people_counter/`), the crew is scoped to that
  folder and never crawls the parent. Pass `-p <dir>` to force a scope.
- **Per-agent activity log** — instead of CrewAI's firehose you get one tidy
  line per step: `▶ Agent 2: Scope Analyst is starting work...`,
  `· Scope Analyst used tool: read_file`, `✔ Scope Analyst completed its work.`
  (in `tracker.py`, via CrewAI's event bus).
- **Sessions** live in `~/.crew_agent_sessions/`; continuing one feeds a
  condensed recap of earlier turns back as context.

## Run it

```bash
pip install -r requirements.txt          # crewai, python-dotenv

python3 main.py -p ~/Desktop/some_repo "fix the bug in calc.py, add should return a + b"
python3 main.py -p ~/Desktop/app "add a method to edit rfid in ChargePointsController"
```

Edits are applied to the target project directly, and the original of every
edited file is copied to `~/.crew_agent_backups/<project>/<timestamp>/` first.

## Configuration

Point it at a different gateway/model with environment variables (or a `.env`
file next to `main.py`):

```
FREELLM_API_KEY=...
FREELLM_BASE_URL=http://localhost:3001/v1
FREELLM_MODEL=auto
```

## How it differs from the original `~/Desktop/agent`

CrewAI gives you the agent orchestration for free, which makes this version much
shorter and easier to follow. In exchange it gives up some of the original's
tight control: there is no token-packing/optimizer, no hard scope enforcement,
and no oscillation/loop safeguards — the agents run more autonomously. Use this
one to understand and extend; use the original when you need the token thrift and
the mechanical guarantees.
