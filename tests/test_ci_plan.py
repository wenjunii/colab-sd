import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import ci_plan


class CICheckSelection(unittest.TestCase):
    def test_documentation_skips_installations_but_unknown_files_still_run_cpu_checks(self):
        plan = ci_plan.plan_checks(['README.md', 'docs/benchmarks/run.json', 'docs/example.png'])
        self.assertFalse(any(plan[key] for key in ('cpu', 'windows', 'ltx', 'waypoint')))
        self.assertTrue(ci_plan.plan_checks(['future-runtime/settings.dat'])['cpu'])

    def test_normal_bridge_changes_keep_cpu_tests_without_reinstalling_runtime_stacks(self):
        plan = ci_plan.plan_checks(['colab_server.py', 'tests/test_streaming.py', 'StreamDiffusion_Colab_Server.ipynb'])
        self.assertTrue(plan['cpu'])
        self.assertFalse(any(plan[key] for key in ('windows', 'ltx', 'waypoint')))

    def test_runtime_changes_select_the_matching_windows_stack_and_shared_imports(self):
        krea = ci_plan.plan_checks(['krea_quant.py'])
        self.assertEqual(krea['windows_matrix']['include'], [{'backend': 'krea', 'python': '3.11'}])
        self.assertFalse(krea['ltx'])
        ltx = ci_plan.plan_checks(['ltx_runtime.py'])
        self.assertEqual(ltx['windows_matrix']['include'], [{'backend': 'ltx', 'python': '3.13'}])
        self.assertTrue(ltx['ltx'])
        shared = ci_plan.plan_checks(['krea_setup.py'])
        self.assertEqual(len(shared['windows_matrix']['include']), 2)
        self.assertTrue(shared['ltx'])

    def test_shared_installers_policy_and_test_dependency_changes_select_all_checks(self):
        for path in ('waypoint_setup.py', 'colab_setup.py', 'workstation.py',
                     'requirements-test.txt', 'ci_plan.py', '.github/workflows/checks.yml'):
            with self.subTest(path=path):
                plan = ci_plan.plan_checks([path])
                self.assertTrue(all(plan[key] for key in ('cpu', 'windows', 'ltx', 'waypoint')))
                self.assertEqual(len(plan['windows_matrix']['include']), 2)

    def test_manual_run_selects_all_without_requiring_merge_history(self):
        with tempfile.TemporaryDirectory(prefix='colab-sd-ci-plan-') as directory:
            output = Path(directory).resolve() / 'outputs.txt'
            with patch.dict(os.environ, {'GITHUB_EVENT_NAME': 'workflow_dispatch', 'GITHUB_OUTPUT': str(output)}), \
                    patch.object(ci_plan, 'changed_paths', side_effect=AssertionError('No Git query needed')), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ci_plan.main([]), 0)
            values = {name: json.loads(value) for name, value in
                      (line.split('=', 1) for line in output.read_text().splitlines())}
            self.assertTrue(all(values[key] for key in ('cpu', 'windows', 'ltx', 'waypoint')))
            self.assertEqual(len(values['windows_matrix']['include']), 2)

    def test_missing_merge_history_fails_instead_of_publishing_skip_outputs(self):
        with tempfile.TemporaryDirectory(prefix='colab-sd-ci-plan-') as directory:
            output = Path(directory).resolve() / 'outputs.txt'
            with patch.dict(os.environ, {'GITHUB_EVENT_NAME': 'pull_request', 'GITHUB_OUTPUT': str(output)}), \
                    patch.object(ci_plan, 'changed_paths', side_effect=subprocess.CalledProcessError(128, ['git'])), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    ci_plan.main([])
            self.assertEqual(stopped.exception.code, 1)
            self.assertFalse(output.exists())

    def test_real_merge_compares_the_entire_pr_to_its_target_and_classifies_renames(self):
        with tempfile.TemporaryDirectory(prefix='colab-sd-ci-merge-') as directory:
            root = Path(directory).resolve()
            environment = os.environ.copy()
            environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull,
                               GIT_CONFIG_NOSYSTEM='1', PYTHONDONTWRITEBYTECODE='1')

            def git(*arguments):
                result = subprocess.run(['git', '-c', 'user.name=CI fixture',
                    '-c', 'user.email=ci-fixture@example.invalid', '-c', 'commit.gpgSign=false',
                    '-c', 'core.hooksPath=' + str(root / 'no-hooks'), *arguments], cwd=root,
                    env=environment, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return result.stdout.decode().strip()

            git('init', '--template=', '--initial-branch=main')
            (root / 'krea_setup.py').write_text('fixture = True\n')
            git('add', '.')
            git('commit', '-m', 'base')
            git('switch', '-c', 'feature')
            git('mv', 'krea_setup.py', 'renamed_setup.py')
            git('commit', '-m', 'rename installer')
            (root / 'README.md').write_text('documentation\n')
            git('add', '.')
            git('commit', '-m', 'later documentation commit')
            git('switch', 'main')
            (root / 'target-only.txt').write_text('target branch change\n')
            git('add', '.')
            git('commit', '-m', 'target advanced')
            git('merge', '--no-ff', 'feature', '-m', 'PR merge fixture')
            paths = ci_plan.changed_paths(root)
            self.assertEqual(set(paths), {'krea_setup.py', 'renamed_setup.py', 'README.md'})
            self.assertEqual(len(ci_plan.plan_checks(paths)['windows_matrix']['include']), 2)
