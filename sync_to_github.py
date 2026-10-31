"""Build, validate, and sync without shell interpolation or unchecked failures."""
import argparse
import datetime
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

PROJECT_DIR = Path(__file__).resolve().parent


class SyncError(RuntimeError):
    pass


def run_cmd(command, root=PROJECT_DIR, check=True):
    result = subprocess.run(command, cwd=root, capture_output=True, text=True,
                            encoding='utf-8', errors='replace')
    if check and result.returncode:
        raise SyncError(f'{shlex.join(str(part) for part in command)} failed:\n{result.stderr.strip() or result.stdout.strip()}')
    return result


def build_and_check(root=PROJECT_DIR):
    for script in ('build_clean_toe.py', 'generate_notebook.py'):
        result = run_cmd([sys.executable, str(root / script)], root)
        if result.stdout.strip():
            print(result.stdout.strip())
    print('Running project tests...', flush=True)
    tests = run_cmd([sys.executable, '-m', 'unittest', 'discover', '-s', str(root / 'tests'), '-v'], root)
    summary = [line.strip() for line in (tests.stdout + '\n' + tests.stderr).splitlines()
               if line.startswith('Ran ') or line.strip() == 'OK']
    if summary:
        print('\n'.join(summary))
    run_cmd([sys.executable, str(root / 'generate_notebook.py'), '--check'], root)
    run_cmd(['git', 'diff', '--check'], root)


def sync(message, root=PROJECT_DIR, check_only=False, dry_run=False, merge=False, admin=False, pr_body_file=None):
    root = Path(root).resolve()
    if admin and not merge:
        raise SyncError('--admin requires --merge')
    if dry_run:
        print('Dry run: build both TouchDesigner archives; regenerate notebook; run tests and notebook checks.')
        if not check_only:
            print('Then stage changes, commit the supplied message as one argument, and push the current branch.')
            print('If main is protected, create a codex/sync-* branch and a pull request.')
            print('Merge the PR only when --merge is specified; bypass protection only with --admin too.')
        return None
    if pr_body_file is not None:
        pr_body_file = Path(pr_body_file).expanduser().resolve()
        if not pr_body_file.is_file():
            raise SyncError(f'PR description file does not exist: {pr_body_file}')
        pr_body = pr_body_file.read_text(encoding='utf-8')
    else:
        pr_body = None
    build_and_check(root)
    if check_only:
        print('Build and validation passed. Git history and remotes were not changed.')
        return None
    branch = run_cmd(['git', 'branch', '--show-current'], root).stdout.strip()
    if not branch:
        raise SyncError('Cannot sync a detached checkout. Switch to a branch first.')
    status = run_cmd(['git', 'status', '--porcelain'], root).stdout.strip()
    if status:
        run_cmd(['git', 'add', '--all'], root)
        staged = run_cmd(['git', 'diff', '--cached', '--quiet'], root, check=False)
        if staged.returncode not in (0, 1):
            raise SyncError(staged.stderr or 'Could not inspect staged changes')
        if staged.returncode == 1:
            run_cmd(['git', 'commit', '--message', message], root)
    # A previous failed push may have left a local commit with a clean working
    # tree. Always retry the push instead of claiming everything is synced.
    pushed = run_cmd(['git', 'push', '-u', 'origin', branch], root, check=False)
    if pushed.returncode == 0:
        print(f'Synced branch {branch}.')
        return None
    detail = (pushed.stderr + pushed.stdout).lower()
    protected = any(marker in detail for marker in ('gh006', 'gh013', 'protected branch', 'repository rule'))
    if branch != 'main' or not protected:
        raise SyncError('Push failed; the local commit is preserved.\n' + (pushed.stderr or pushed.stdout))
    sync_branch = 'codex/sync-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    run_cmd(['git', 'switch', '-c', sync_branch], root)
    run_cmd(['git', 'push', '-u', 'origin', sync_branch], root)
    with tempfile.TemporaryDirectory(prefix='colab-sd-pr-') as directory:
        body_path = Path(directory).resolve() / 'body.txt'
        assert body_path.parent.is_relative_to(Path(tempfile.gettempdir()).resolve())
        body_path.write_text(pr_body if pr_body is not None else
                             message + '\n\nValidation: rebuilt .toe/.tox, regenerated the Colab notebook, and passed the project checks.\n',
                             encoding='utf-8')
        created = run_cmd(['gh', 'pr', 'create', '--title', message, '--body-file', str(body_path),
                           '--base', 'main', '--head', sync_branch], root)
    url = created.stdout.strip()
    if not url.startswith('https://'):
        raise SyncError('PR creation did not return a valid URL. The sync branch is preserved.')
    if not merge:
        print(f'Created pull request: {url}\nCurrent branch: {sync_branch}')
        return url
    command = ['gh', 'pr', 'merge', url, '--merge', '--delete-branch']
    if admin:
        command.append('--admin')
    run_cmd(command, root)
    run_cmd(['git', 'switch', 'main'], root)
    run_cmd(['git', 'pull', '--ff-only', 'origin', 'main'], root)
    print(f'Merged and synced: {url}')
    return url


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('message', nargs='?', default='Update ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
    parser.add_argument('--check', action='store_true', help='Build and validate without staging, committing, or pushing')
    parser.add_argument('--dry-run', action='store_true', help='Show the workflow without running commands or writing files')
    parser.add_argument('--merge', action='store_true', help='Merge a protected-branch PR after creation')
    parser.add_argument('--admin', action='store_true', help='Explicitly bypass protection when merging (requires --merge)')
    parser.add_argument('--pr-body-file', type=Path, help='UTF-8 description to use if a protected-branch PR is needed')
    args = parser.parse_args(argv)
    try:
        sync(args.message, check_only=args.check, dry_run=args.dry_run, merge=args.merge, admin=args.admin,
             pr_body_file=args.pr_body_file)
    except (OSError, SyncError) as exc:
        print(f'Sync stopped: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())
