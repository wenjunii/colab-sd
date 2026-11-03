# Development resource cleanup

The user wants disposable development resources cleaned up automatically after work finishes.

- Reuse an existing isolated CPU test environment for `requirements-test.txt`. Do not install CUDA or TensorRT just to run CPU checks, or create another environment when the existing one works.
- Put one-off test environments, probes, fixture data, helper scripts, and logs inside a single `tempfile.TemporaryDirectory` for that task. Use context managers or `try/finally` so the task's scratch is removed after success or failure. Keep useful failure output in the command result before cleanup.
- Stop and wait for subprocesses owned by the task before its scratch context exits. Do not remove an environment or directory while a helper still uses it, and do not stop unrelated applications.
- Before recursive cleanup or moving files on Windows, verify the resolved absolute source and destination are inside the task's intended temporary folder. Never scan general Temp directories and delete by project-name glob or age alone.
- Preserve reusable CPU test environments, `.workstation`, model downloads, TensorRT caches, captures, TouchDesigner backups, deliverable screenshots, and built `.toe`/`.tox` files. Keep reusable caches outside disposable scratch. Only a cache explicitly created as disposable test data belongs in the task's cleanup scope.
- Existing files, review bundles, and resources created outside the current task's owned scratch require separate inspection. Do not adopt them into automatic cleanup or bypass an approval rejection with another deletion mechanism.

`python sync_to_github.py --check` and normal sync runs give their build/test subprocesses a private temporary folder, disable new Python bytecode writes in those subprocesses, and remove their owned scratch when the checks finish, including on failure. This does not install an OS-wide cleanup task.
