"""Audit the installed CPU environment, including official Torch +cpu wheels."""
from importlib.metadata import distributions
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from packaging.utils import canonicalize_name
from packaging.version import Version


def installed_requirements(installed=None):
    requirements = {}
    for dist in distributions() if installed is None else installed:
        name = canonicalize_name(dist.metadata['Name'])
        version = Version(dist.version)
        if name == 'torch' and version.local == 'cpu':
            version = Version(version.public)
        elif version.local:
            raise ValueError(f'Cannot audit unsupported local build: {name}=={version}')
        requirements[name] = str(version)
    if not requirements:
        raise ValueError('No installed packages found to audit')
    return requirements


def validate_report(report, expected):
    dependencies = report['dependencies']
    actual = {canonicalize_name(dep['name']) for dep in dependencies}
    if actual != set(expected):
        raise ValueError('Audit results do not cover the installed package inventory')
    for dep in dependencies:
        name = canonicalize_name(dep['name'])
        if 'skip_reason' in dep:
            raise ValueError(f"Audit skipped {name}: {dep['skip_reason']}")
        if Version(dep['version']) != Version(expected[name]):
            raise ValueError(f'Audit returned a different version for {name}')
        if dep['vulns']:
            identifiers = ', '.join(vuln['id'] for vuln in dep['vulns'])
            raise ValueError(f"{name}=={dep['version']}: {identifiers}")


def main():
    try:
        requirements = installed_requirements()
        with tempfile.TemporaryDirectory(prefix='colab-sd-audit-') as directory:
            inventory = Path(directory) / 'requirements.txt'
            inventory.write_text(''.join(f'{name}=={version}\n' for name, version in sorted(requirements.items())),
                                 encoding='utf-8')
            result = subprocess.run(
                [sys.executable, '-m', 'pip_audit', '--requirement', str(inventory),
                 '--no-deps', '--disable-pip', '--format', 'json'],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        report = json.loads(result.stdout)
        validate_report(report, requirements)
        if result.returncode:
            raise ValueError(f'pip-audit failed with exit code {result.returncode}')
        print(f'Audited all {len(requirements)} installed packages, including the Torch CPU base release.')
        return 0
    except (ValueError, KeyError, OSError, subprocess.TimeoutExpired) as error:
        print(f'Dependency audit failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
