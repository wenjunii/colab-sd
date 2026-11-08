"""Select pull-request checks locally; manual workflow runs exercise every stack."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess


WINDOWS_STACKS = ({'backend': 'krea', 'python': '3.11'},
                  {'backend': 'ltx', 'python': '3.13'})
FULL_CHECK_FILES = {'.github/workflows/checks.yml', 'ci_plan.py', 'requirements-test.txt'}
SHARED_INSTALLERS = {'colab_setup.py', 'waypoint_setup.py', 'workstation.py'}
LTX_SHARED = {'krea_setup.py', 'krea_weights.py'}


def documentation_only(path):
    path = PurePosixPath(path)
    return (path.suffix.lower() in ('.md', '.rst')
            or (path.parts[0] == 'docs' and path.suffix.lower() in
                ('.png', '.jpg', '.jpeg', '.svg', '.json', '.csv')))


def plan_checks(paths, full=False):
    paths = set(paths)
    full = full or bool(paths & FULL_CHECK_FILES) or any(
        path.startswith('.github/actions/') for path in paths)
    shared = full or bool(paths & SHARED_INSTALLERS)
    krea = shared or any(path.startswith('krea_') and path.endswith('.py') for path in paths) or 'check_krea_imports.py' in paths
    ltx = (shared or bool(paths & LTX_SHARED)
           or any(path.startswith('ltx_') and path.endswith('.py') for path in paths)
           or 'check_ltx_imports.py' in paths)
    matrix = {'include': [stack for stack, enabled in zip(WINDOWS_STACKS, (krea, ltx)) if enabled]}
    return {'cpu': full or any(not documentation_only(path) for path in paths),
            'waypoint': shared or 'tests/test_waypoint_setup.py' in paths,
            'ltx': ltx, 'windows': krea or ltx, 'windows_matrix': matrix}


def changed_paths(root=Path('.')):
    # checkout@v7 leaves a PR merge commit at HEAD. Its first parent is the
    # target branch, so this includes the full PR, not just the last PR commit.
    # Disable rename detection to classify both sides of moved/deleted files.
    result = subprocess.run(['git', 'diff', '--no-renames', '--name-only', '-z', 'HEAD^1', 'HEAD'],
                            cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return [os.fsdecode(name) for name in result.stdout.split(b'\0') if name]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Select every check without querying Git history')
    args = parser.parse_args(argv)
    full = args.full or os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch'
    try:
        paths = [] if full else changed_paths()
    except (OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Cannot determine changed files; checks were not silently skipped: {error}\n')
    plan = plan_checks(paths, full=full)
    print(json.dumps({'changed_files': len(paths), 'full': full, **plan}, indent=2))
    output = os.environ.get('GITHUB_OUTPUT')
    if output:
        with Path(output).open('a', encoding='utf-8') as handle:
            for name, value in plan.items():
                handle.write(f'{name}={json.dumps(value, separators=(",", ":"))}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
