"""Regression checks for Colab's missing-ensurepip and partially created venv."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import waypoint_setup as installer


def result(code=0, output=''):
    return subprocess.CompletedProcess([], code, stdout=output)


class WaypointSetupChecks(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix='colab-sd-bootstrap-fixture-')
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.environment = self.root / 'venv'
        self.python = self.environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        self.state = json.dumps({'prefix': str(self.environment), 'base': sys.base_prefix,
                                 'version': list(sys.version_info[:2])})

    def partial_environment(self):
        self.python.parent.mkdir(parents=True)
        self.python.touch()
        (self.environment / 'pyvenv.cfg').write_text('include-system-site-packages = false\n')

    def test_new_environment_bootstraps_pip_without_ensurepip(self):
        with patch.object(installer.venv, 'EnvBuilder') as builder, \
                patch.object(installer.subprocess, 'run', side_effect=[
                    result(output=self.state), result(1, 'No module named pip'),
                    result(output='pip installed'), result(output='pip 26.3')]) as run:
            python, changed = installer.ensure_environment(self.environment)
        builder.assert_called_once_with(with_pip=False)
        self.assertEqual(python, self.python)
        self.assertTrue(changed)
        bootstrap = run.call_args_list[2].args[0]
        self.assertEqual(bootstrap[:5], [sys.executable, '-m', 'pip', '--python', str(self.python)])
        self.assertNotIn('ensurepip', repr(run.call_args_list))

    def test_partial_environment_with_python_but_no_pip_is_repaired_in_place(self):
        self.partial_environment()
        sentinel = self.environment / 'existing-package.txt'
        sentinel.write_text('keep')
        with patch.object(installer.venv, 'EnvBuilder') as builder, \
                patch.object(installer.subprocess, 'run', side_effect=[
                    result(output=self.state), result(1, 'No module named pip'),
                    result(), result(output='pip 26.3')]):
            _, changed = installer.ensure_environment(self.environment)
        builder.assert_called_once_with(with_pip=False)
        self.assertTrue(changed)
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_working_environment_is_reused_without_bootstrap(self):
        self.partial_environment()
        with patch.object(installer.venv, 'EnvBuilder') as builder, \
                patch.object(installer.subprocess, 'run', side_effect=[
                    result(output=self.state), result(output='pip 26.2.2')]) as run:
            _, changed = installer.ensure_environment(self.environment)
        builder.assert_not_called()
        self.assertFalse(changed)
        self.assertEqual(run.call_count, 2)

    def test_bootstrap_failure_includes_actual_child_error(self):
        self.partial_environment()
        with patch.object(installer.venv, 'EnvBuilder'), \
                patch.object(installer.subprocess, 'run', side_effect=[
                    result(output=self.state), result(1, 'No module named pip'),
                    result(1, 'Package index connection refused')]):
            with self.assertRaisesRegex(RuntimeError, 'Package index connection refused'):
                installer.ensure_environment(self.environment)

    def test_successful_bootstrap_still_requires_working_target_pip(self):
        self.partial_environment()
        with patch.object(installer.venv, 'EnvBuilder'), \
                patch.object(installer.subprocess, 'run', side_effect=[
                    result(output=self.state), result(1), result(), result(1, 'pip still missing')]):
            with self.assertRaisesRegex(RuntimeError, 'Isolated pip verification.*failed'):
                installer.ensure_environment(self.environment)

    def test_base_interpreter_cannot_be_used_as_install_target(self):
        self.partial_environment()
        state = json.dumps({'prefix': sys.base_prefix, 'base': sys.base_prefix,
                            'version': list(sys.version_info[:2])})
        with patch.object(installer.subprocess, 'run', return_value=result(output=state)) as run:
            with self.assertRaisesRegex(RuntimeError, 'not an isolated environment'):
                installer.ensure_environment(self.environment)
        self.assertEqual(run.call_count, 1)

    def test_valid_install_marker_avoids_reinstalling_gpu_packages(self):
        signature = {'python': list(sys.version_info[:2]), 'dependencies': list(installer.DEPENDENCIES)}
        (self.root / 'installed.json').write_text(json.dumps(signature))
        (self.root / ('cloudflared.exe' if os.name == 'nt' else 'cloudflared')).touch()
        with patch.object(installer, 'ensure_environment', return_value=(self.python, False)), \
                patch.object(installer.subprocess, 'run', return_value=result()) as run:
            installer.setup(self.root)
        self.assertFalse(any('install' in call.args[0] for call in run.call_args_list))
