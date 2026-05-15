---
name: import-repo
description: Import a local git repository (or any local directory tree) into the user's Qontext workspace via the qontext-ai MCP server. Use this skill whenever the user wants to bulk-load a repo, codebase, project, or folder of source files into Qontext, their workspace, or their vault. Triggers on phrases like "import repo to qontext", "ingest my codebase", "upload this project to my workspace", "load files into the vault", "qontext repo import", "put this repo into qontext", or anything similar. Trigger even if the user does not say "Qontext" by name — if they mention importing/ingesting/uploading a directory of files in the context of a workspace, vault, or MCP server that handles knowledge, use this skill.
allowed-tools: AskUserQuestion, Bash, Read, Glob, mcp__qontext-ai__qontext_ls, mcp__qontext-ai__qontext_find, mcp__qontext-ai__qontext_mkdir, mcp__qontext-ai__qontext_write, mcp__qontext-ai__qontext_rm
---

# import-repo

Bulk-import a local repository or directory tree into the user's Qontext workspace via the `qontext-ai` MCP server.

> Note on naming: the user installed the qontext-ai MCP server under whatever local name they chose — could be `qontext-ai`, `qontext`, `qontext-ai-mcp`, `quontext`, anything. The detection step below resolves the real name from the user's MCP config; use that resolved name as the tool prefix (i.e., `mcp__<resolved>__qontext_ls`, etc.) for every MCP call in this skill. In anything you say to the user, refer to the server as **qontext-ai** regardless of their local naming.

The flow is interactive: scan the repo, ask the user a small number of focused questions one at a time, then execute the import in a single confirmed pass.

This skill is intended to be **single-use**. After a fully successful import, delete the skill's directory from disk so it won't keep triggering on future prompts. On partial failure, leave it in place so the user can re-run after fixing the underlying issues.

## Inputs

The user invokes the skill with a repo reference. Accept either:

- A local path to a directory — absolute, relative, or with `~` expansion. This is the preferred form.
- A git URL — clone it shallow first into `/tmp/qontext-import-<short-id>/` and treat that path as the source.

If the user did not include any path, ask for one. Do not guess.

---

## Step 1 — Verify prerequisites

### 1a. Detect, then verify the qontext-ai MCP

The user may have installed the qontext-ai MCP server under any local name. Don't assume `qontext-ai`. Resolve the actual name from their MCP config first.

#### Step 1a.i — Find the qontext-ai server

Run `claude mcp list` via Bash and identify any entry whose configured URL points at the production qontext.ai endpoint:

```bash
claude mcp list
```

Look for a line whose URL or command-args contain `api.qontext.ai/mcp` (the prod endpoint). The MCP name is the first column / the key. Common shapes you'll see:

- `<name>: http(s) - https://api.qontext.ai/mcp`
- `<name>: stdio - npx mcp-remote https://api.qontext.ai/mcp`

Record the resolved name (e.g., `qontext`, `qontext-ai`, `qontext-ai-mcp`) — call it `<SERVER>`. Every MCP tool call below will use `mcp__<SERVER>__qontext_*`.

#### Step 1a.ii — Handle the three branches

- **No server found pointing at `api.qontext.ai/mcp`** → Install it yourself via Bash, then inform the user and stop. Do **not** ask the user to run the command.

  ```bash
  claude mcp add qontext-ai npx mcp-remote https://api.qontext.ai/mcp
  ```

  Then tell the user:

  > I've added the qontext-ai MCP server (proxied through `mcp-remote` to `https://api.qontext.ai/mcp`). When the OAuth flow pops up in your browser, complete it. Once you're authenticated, re-run this skill and I'll continue the import.

  Stop here. The MCP client needs to pick up the new server and complete OAuth before any tool calls will succeed.

- **Server found** → Probe it with `mcp__<SERVER>__qontext_ls` using `{ "limit": 1 }` and no `path`. Three sub-outcomes:

  - **Success** → MCP is installed and authenticated. Continue to step 1b.

  - **Auth-shaped error** (401, "not authenticated", "auth required", "token expired") → The server is configured but the user's OAuth token is missing or expired. Tell the user:

    > Found qontext-ai installed as `<SERVER>` but it's not authenticated. Open `/mcp` in Claude Code, select `<SERVER>`, and complete the OAuth flow. Then re-run this skill.

    Stop. Do not try to silently re-install — that would clobber their existing config under a different name.

  - **Any other error** (network, server 5xx, malformed response) → Surface the error verbatim to the user and stop. They'll need to investigate before this skill can do anything useful.

### 1b. Verify the repo path

Use `Bash` to confirm the path exists and is a directory:

```bash
test -d "$REPO_PATH" && echo OK
```

If it's a directory but not a git repo (`.git/` absent), proceed anyway — non-git directories are also valid imports. If the user gave a URL, clone first:

```bash
git clone --depth 1 "$URL" /tmp/qontext-import-$(date +%s)
```

---

## Step 2 — Scope, then analyze, then ask

Order matters: ask about scope **before** the deeper analysis, so the rest of the questions only reflect files the user actually wants to import. If you analyze the whole repo first, the file-type / dot-folder / sensitive-file / oversized-file findings will be polluted by files outside the user's chosen scope, and you'll end up asking decisions about files we're going to skip anyway.

### Quick scan (just enough for the scope question)

Before any question, do a *light* scan — only what's needed to power 2a:

```bash
# Top-level entries in the repo (immediate children of root, excluding .git)
ls -1 "$REPO_PATH" | grep -vE '^\.git$' | sort

# Whole-repo file count + total bytes — used for the size warning in 2a
find "$REPO_PATH" -type f -not -path '*/\.git/*' | wc -l
du -sh "$REPO_PATH" --exclude=.git

# Per-top-level-dir file count (useful for the follow-up question in 2a)
for d in "$REPO_PATH"/*/; do
  [ -d "$d" ] && printf '%s\t%s\n' "$(basename "$d")" \
    "$(find "$d" -type f -not -path '*/\.git/*' | wc -l)"
done
```

Keep the top-level entries, their file counts, and the whole-repo size in memory.

### Ask the user, using `AskUserQuestion`, one question at a time

Skip any question whose answer is already determined by what you've found so far (e.g., don't ask about submodules if there are none). The order below is the recommended order.

#### 2a. Scope

Always ask. If the whole-repo file count exceeds **500** or total bytes exceed **50 MB**, embed a warning in the question's `description`: *"Heads up: full repo is N files (X MB). Large imports may be slow."*

- "Full repo" *(recommended)*
- "Specific subfolders only" — requires a follow-up

**If the user picks "Specific subfolders only", immediately ask a follow-up question to pick which ones.** Use the per-top-level-dir file counts you scanned earlier:

- If 2–4 top-level dirs exist (the common case): present each as a `multiSelect: true` option in `AskUserQuestion`. Include the file count per dir in each option's `description`, e.g. *"src/ — 142 files"*, *"docs/ — 12 files"*.
- If more than 4 exist: present the 3 with the most files as `multiSelect: true` options. The "Other" slot (always available on `AskUserQuestion`) lets the user type a comma-separated free-text list (e.g. `src,docs,scripts`). Sub-paths under a top-level dir are allowed too (e.g. `src/api,docs/guides`) — don't restrict to top-level.

Record the final list of selected paths as **`<SCOPE>`** — one or more paths under the repo root. If the user picked "Full repo", `<SCOPE>` is the repo root itself.

### Deep scan (within `<SCOPE>` only)

Now run the rest of the analysis, restricted to the paths in `<SCOPE>`. Every subsequent question below uses **only** files inside `<SCOPE>`, so we don't surface decisions about files we won't touch.

```bash
# In a shell loop where $SCOPE_PATHS is space-separated absolute paths under $REPO_PATH
for p in $SCOPE_PATHS; do
  # Extension breakdown
  find "$p" -type f -not -path '*/\.git/*' \
    | sed -E 's/.*\.([^./]+)$/\1/' \
    | sort | uniq -c | sort -rn

  # Dot-folders within scope (besides .git)
  find "$p" -type d -name '.*' -not -path '*/\.git*' -mindepth 1

  # Submodules whose path is inside this scope path
  # (Cross-reference $REPO_PATH/.gitmodules entries with $p, or use `git submodule status`)

  # Sensitive files within scope
  find "$p" -type f \
    \( -name '.env*' -o -name '*.pem' -o -name 'id_rsa' \
       -o -iname '*credentials*' -o -name '*.key' \) \
    -not -path '*/\.git/*'

  # Oversized files (>45 KB) within scope
  find "$p" -type f -not -path '*/\.git/*' -size +45k
done
```

Aggregate the results across all selected paths (sum the extension counts; union the dot-folder, submodule, sensitive, and oversized lists). Keep this aggregate in memory for the remaining questions.

#### 2b. Target Qontext path

Default to `/<repo-name-slug>` (lowercase, dashes for spaces/underscores). Allow free-text override.

Once chosen, call `qontext_ls` on the target to see whether it already has content. Note: if the folder doesn't exist yet, `qontext_ls` may error with not-found — that's fine, treat as "empty, will create".

If the target already has content, ask:

- "Abort" *(recommended if unsure)*
- "Overwrite matching paths" — `qontext_write` will replace existing files at the same paths
- "Skip existing" — files that already exist in Qontext get left alone

#### 2c. Non-`.md` files

Ask if any non-`.md` files were found. Show the top 10 extensions with counts in the question's `description`, e.g. *"Found: .py × 142, .ts × 88, .json × 12, .css × 9, .yaml × 4, ..."*.

- "Ignore (recommended)" — only existing `.md` files get imported; everything else skipped
- "Import as `.md` with fenced code block" — wrap each file's content in a markdown fence, language tag inferred from extension (see "Extension → language" table below)
- Other (free text) — let the user describe a custom rule

#### 2d. Hidden / dot-folders

Only ask if the scan found dot-folders beyond `.git`. List them in the description.

- "Skip all (recommended)"
- "Include selected" — multi-select which ones to include

Add this note to the description: *"Qontext does not allow folder names starting with a dot — included dot-folders will be renamed (e.g., `.github` → `dot-github`)."*

#### 2e. Submodules

Only ask if the deep scan found at least one submodule whose path is inside `<SCOPE>`. A `.gitmodules` file at the repo root alone is not enough — if none of its entries fall within the user's chosen scope, skip this question.

- "Skip (recommended)"
- "Store pointer text" — write a small `.md` placeholder at each submodule path containing the submodule URL + commit
- "Recurse" — walk into each submodule and import its files too

#### 2f. Sensitive files

Only ask if the scan found any. List them in the description.

- "Skip all (recommended)"
- "Include anyway" — caution them: these files often contain secrets

#### 2g. Size limit

Only ask if any oversized files were found. Surface the count and total bytes.

- "Skip oversized files (recommended)" — the MCP has a hard 45 KB per-file cap; these will not be written
- "Split into a sibling folder" — write a stub `.md` in place with a pointer, and dump the (truncated) content into a `_oversized/` sibling folder. Lower priority — only implement this branch if the user explicitly picks it.

#### 2h. Final confirm

After all questions are answered, summarize the plan in plain text. Example:

```
Import plan
  Source:        /Users/nikita/Repositories/my-repo  (342 files, 18.2 MB)
  Target Qontext:    /my-repo
  Will import:   312 files
  Skipping:      28 files (24 binary, 3 oversized, 1 sensitive)
  Non-.md files: wrap in fenced code blocks
  Dot-folders:   skip all
  Submodules:    n/a
```

Then ask one final yes/no via `AskUserQuestion`: *"Proceed with import?"*

- "Proceed"
- "Cancel"

Do **not** start writing until the user picks Proceed.

---

## Step 3 — Migrate

### Folder creation (BFS, top-down)

`qontext_mkdir` does **not** have `-p` semantics. Every folder's parent must already exist before you create the child. Walk the import tree breadth-first and `mkdir` each level in turn.

Idempotency note: if `qontext_mkdir` fails because the folder already exists, log and continue — that's fine for re-runs. Distinguish "already exists" errors from other errors so you don't swallow real problems.

#### Folder name sanitization (before each `mkdir`)

Qontext rejects certain folder names. Apply these rules in order:

1. **Leading dot** → replace with `dot-`. `.github` becomes `dot-github`, `.vscode` becomes `dot-vscode`.
2. **Trailing `.md`** → strip it. (`docs.md` as a folder name would be rejected; rename to `docs`.)
3. **Forbidden characters** (`; $ | & \ < > * ? [ ] { } : ` and backtick) → replace with `-`.
4. **Empty after sanitization** → use `folder`.

If sanitization causes a collision with an existing sibling (e.g., the source has both `.github` and `dot-github`), append `-1`, `-2`, etc. to the second one to disambiguate.

### File writing

For each file (BFS order within each folder):

1. **Binary detection.** Read the first 8 KB. If any NUL byte (`0x00`) is present, treat as binary and **skip** (log reason).
2. **Size check.** If size > 45 KB, **skip** (log reason) — or apply the user's split-folder choice from 2g if they picked it.
3. **Compute Qontext path:**
   - Take the path relative to the import root
   - Apply folder sanitization to each directory segment
   - If the filename already ends in `.md`, keep it. Otherwise, append `.md` (`foo.py` → `foo.py.md`, `Dockerfile` → `Dockerfile.md`).
   - Sanitize the filename itself for forbidden characters (same set as folders)
   - Prepend the target Qontext path
   - Important: files cannot live at root. If the user picked a target of `/`, refuse and prepend a `/<repo-name>` wrapper instead.
4. **Compose the body:**
   - If the source file already ended in `.md`, use its content as-is (no wrapping).
   - Otherwise, wrap in a fenced code block. To avoid fence collision: scan the file for the longest run of consecutive backticks; use that run length **+ 1** as the fence length (minimum 3 backticks). Open with the language tag (see table below), close with the same number of backticks.

     Example for a `.py` file:

     ````
     ```python
     <file content>
     ```
     ````

5. **Write** via `qontext_write` with a commit message like `"import: <repo-relative-path> from <repo-name>"`.
6. **Retry on failure once.** If the retry also fails, record a permanent failure for this file and continue. Do **not** abort the run on a single file's failure.
7. **Progress.** Print `[N/total] <qontext-path>` between writes so the user sees motion. Keep it on one line of output per file; don't pad with extra prose.

### Extension → language tag (for fenced wraps)

| Extension(s) | Language tag |
|---|---|
| `.py`, `.pyi` | `python` |
| `.ts`, `.tsx` | `typescript` |
| `.js`, `.jsx`, `.mjs`, `.cjs` | `javascript` |
| `.go` | `go` |
| `.rs` | `rust` |
| `.java` | `java` |
| `.kt`, `.kts` | `kotlin` |
| `.rb` | `ruby` |
| `.sh`, `.bash`, `.zsh` | `bash` |
| `.json` | `json` |
| `.yaml`, `.yml` | `yaml` |
| `.toml` | `toml` |
| `.xml` | `xml` |
| `.html`, `.htm` | `html` |
| `.css` | `css` |
| `.scss`, `.sass` | `scss` |
| `.sql` | `sql` |
| `.c`, `.h` | `c` |
| `.cpp`, `.hpp`, `.cc`, `.cxx` | `cpp` |
| `.swift` | `swift` |
| `.gradle` | `groovy` |
| `.proto` | `protobuf` |
| `.dockerfile`, filename `Dockerfile` | `dockerfile` |
| `.tf` | `hcl` |
| `.lua` | `lua` |
| `.php` | `php` |
| `Makefile`, `.mk` | `makefile` |
| anything else / no extension | leave the tag blank after the opening fence |

When in doubt for an ambiguous extension (`.h`, `.m`, `.ts`), pick the more common interpretation in modern usage (`.h` → `c`, `.m` → `objectivec`, `.ts` → `typescript`).

### Final summary

After the loop completes, print a summary:

```
Done.
  Imported:  312 files
  Skipped:   28 files
    - binary:    24
    - oversized: 3
    - sensitive: 1
  Failed:    0 files
```

If `Failed > 0`, list each failed file with its Qontext path and error message under the count.

---

## Step 4 — Self-delete (full success only)

If `Failed == 0`, the skill has done its job. Remove it from disk:

```bash
rm -rf ~/.claude/skills/import-repo/
```

Then tell the user:

> Import complete and the import-repo skill has been removed. If you want to import another repo later, reinstall the skill from its source.

If `Failed > 0`, **do not delete**. Tell the user:

> Import finished with some failures (listed above). I've left the skill in place — fix the underlying issues and re-run with "import this repo to qontext". Re-runs are safe: existing files in Qontext are overwritten (or skipped, per your earlier choice).

---

## Qontext naming rules — cheat sheet

These constraints come from the `qontext-ai` MCP. Bake them into every path you construct:

- **Absolute paths only.** Everything starts with `/`.
- **File paths must end in `.md`.**
- **Files cannot live at the root.** Every file must be under at least one folder.
- **Folder names must not start with `.`** (use a `dot-` prefix).
- **Folder names must not end in `.md`** (that's the file suffix).
- **Forbidden characters in any name:** `; $ | & \ < > * ? [ ] { } : ` and backtick (also control characters).
- **`qontext_mkdir` is not recursive** — parents must exist.
- **`qontext_write` overwrites by default** — old content is preserved in commit history (good news for safe re-runs).
- **`qontext_rm` is not recursive** — to clean up a folder, delete every descendant first. This makes mid-run rollback expensive; the skill chooses to log + continue on per-file failures rather than try to roll back.

---

## Failure modes & recovery

- **MCP auth lost mid-run** → abort, tell the user to re-auth, mention which file the run got to. Safe to re-run; previously written files are kept.
- **Network / 5xx on a write** → handled by the retry-once rule. Persistent failure → log and continue.
- **Sanitization collision** → handled by `-1`/`-2` suffixing.
- **Path too long / Qontext rejects path** → log and skip that file; continue.
- **User cancels mid-run** → just stop. Already-written files stay in Qontext. The skill is not deleted.

---

## Notes on tone and output

- During scanning and migration, keep your text output to the user **terse and informative**. One-line progress prints are great. Don't narrate every internal step.
- For the question phase, let `AskUserQuestion` do the talking — keep any preamble between questions to a single sentence.
- The final summary is the only place a multi-line block of prose is appropriate.
