# Working preferences

## Automatic cleanup after tasks

- The user authorizes routine cleanup of disposable files created or downloaded for a task once verification is finished, without asking again. Include one-off validation scripts, scratch files, completed installation archives, abandoned partial downloads, and redundant task copies when they are no longer needed.
- Verify exact targets and dependencies before removal. Limit cleanup to known task artifacts; age or absence from the current workflow alone does not establish that a file is unused.
- When the user retires a workflow, also remove models downloaded exclusively for that workflow after checking for shared dependencies. Preserve active or shared models, required tools and environments, source files, reference inputs, and user outputs unless their removal is authorized.
- Prefer the Recycle Bin when practical. Do not empty the Recycle Bin or permanently remove unrelated files as part of routine cleanup.
- Briefly report material cleanup, approximate size when useful, and whether the files are recoverable. Perform cleanup as part of task completion; this preference does not create a background schedule.

## Development resource cleanup

- Reuse an existing isolated CPU test environment for `requirements-test.txt`. Do not install CUDA or TensorRT just to run CPU checks, or create another environment when the existing one works.
- Put one-off test environments, probes, fixture data, helper scripts, and logs inside a single `tempfile.TemporaryDirectory` for that task. Use context managers or `try/finally` so the task's scratch is removed after success or failure. Keep useful failure output in the command result before cleanup.
- Stop and wait for subprocesses owned by the task before its scratch context exits. Do not remove an environment or directory while a helper still uses it, and do not stop unrelated applications.
- Before recursive cleanup or moving files on Windows, verify the resolved absolute source and destination are inside the task's intended temporary folder. Never scan general Temp directories and delete by project-name glob or age alone.
- Preserve reusable CPU test environments, `.workstation`, active or shared model downloads and TensorRT caches, captures, TouchDesigner backups, deliverable screenshots, and built `.toe`/`.tox` files. Keep reusable caches outside disposable scratch. Only a cache explicitly created as disposable test data belongs in the task's per-run cleanup scope. Removing models from a retired workflow requires a separate dependency check under the preferences above.
- Existing files, review bundles, and resources created outside the current task's owned scratch require separate inspection. Do not adopt them into automatic cleanup or bypass an approval rejection with another deletion mechanism.

`python sync_to_github.py --check` and normal sync runs give their build/test subprocesses a private temporary folder, disable new Python bytecode writes in those subprocesses, and permanently remove only that run's disposable scratch when the checks finish, including on failure. The success message identifies this removal as permanent. This does not install an OS-wide cleanup task.
