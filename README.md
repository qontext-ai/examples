# Qontext AI Examples

Demos and sample projects that use the [Qontext AI](https://qontext.ai) API for retrieval and knowledge grounding. Use them as reference for agents or plugging Qontext into any LLM or workflow.

## What's in this repo

### Examples

- **gemini-agent** ([`examples/gemini-agent`](examples/gemini-agent)) — Agent that answers prompts based on context saved in a Qontext vault. Follow the instructions in our Gemini Docs for the correct implementation.

### Skills

[Claude Code skills](https://docs.claude.com/en/docs/claude-code/skills) that drive Qontext through the `qontext-ai` MCP server.

- **import-repo** ([`skills/import-repo`](skills/import-repo)) — Bulk-import a local git repository (or any directory tree) into a Qontext workspace. Interactive: scans the repo, asks about scope, file-type handling, and sensitive-file treatment, then writes everything via the qontext-ai MCP. Single-use — deletes itself on a fully successful import.

  Install with [`skills.sh`](https://skills.sh):

  ```bash
  npx skills add https://github.com/qontext-ai/examples/tree/main/skills/import-repo
  ```

  Then invoke from Claude Code with something like `"import this repo to qontext: /path/to/my-repo"`.

## Resources
- [Qontext AI](https://qontext.ai)
- [Qontext Docs](https://docs.qontext.ai/get-started)
