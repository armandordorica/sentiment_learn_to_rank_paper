---
name: validate-before-commit
description: Always test and validate code changes before committing or pushing in this repo. Use whenever about to commit, push, stage for commit, or finish a coding change to app.py, scripts/, src/sentiment_ltr/, or notebooks.
---

# Validate Before Commit

## Principle

Always test and validate before committing. Never commit or push untested code.

## Workflow

Before any `git commit` / `git push`, run this checklist:

```
- [ ] Lint the files you changed (ReadLints), fix new errors
- [ ] Validate syntax/import of changed Python (use the project conda env)
- [ ] Run logic tests for the changed behavior (mock network/WRDS if needed)
- [ ] Only after all checks pass: stage, commit, push
```

## How To Validate Here

- **Python interpreter**: use the project env, not the sandbox default (which lacks
  pandas/streamlit):
  `/Users/armandoordoricadelatorre/miniconda/envs/sentiment-ltr-paper/bin/python`
- **Syntax/import check**: `python -c "import ast; ast.parse(open('app.py').read())"`
  and `python -c "import sys; sys.path.insert(0,'src'); import app"`.
- **No live credentials/network?** Mock the boundary (e.g. monkeypatch
  `query_crsp_delisting`, WRDS, or provider calls) and test the surrounding logic
  offline. Treat `FutureWarning` as an error in tests to catch deprecations:
  `warnings.simplefilter("error", FutureWarning)`.
- **Cache/idempotency changes**: assert the "only fetch missing" path — first call
  queries N, repeat call queries 0, adding one item queries exactly 1, `force`
  re-queries.

## Rules

- If a test surfaces an issue (even a warning), fix it and re-run before committing.
- If full validation is impossible in this environment (needs real WRDS/network),
  do every offline check possible, then state clearly what still needs a live run
  and do not claim it passed.
- Do not commit unless the user asked you to; when you do, validation comes first.
