import ast
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import build_clean_toe as builder
import colab_launcher
import generate_notebook
import sync_to_github as syncer


class ProjectToolsChecks(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(prefix='colab-sd-test-')
        self.root = Path(self.workspace.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.addCleanup(self.workspace.cleanup)

    def make_build_sources(self):
        source = self.root / 'colab-sd.toe.dir'
        project = source / 'project1'
        (project / 'StreamDiffusion_Remote').mkdir(parents=True)
        (source / '.build').write_text('2025.33230\n')
        (project / 'StreamDiffusion_Remote.n').write_text('COMP:base\ninputs\n{\n0 noise1\n}\nend\n')
        (self.root / 'colab-sd.toe.toc').write_text('# 4 0 0 0 1\n.build\n')
        executable = self.root / 'toecollapse.exe'
        executable.touch()
        for name in ('colab-sd.toe', 'StreamDiffusion_Remote.tox'):
            (self.root / name).write_bytes(b'previous working output')
        return executable

    def test_failed_second_build_preserves_both_previous_outputs(self):
        executable = self.make_build_sources()

        def compile_archive(command, cwd, **kwargs):
            if command[-1].endswith('.toe'):
                (Path(cwd) / command[-1]).write_bytes(b'x' * 256)
                return SimpleNamespace(returncode=0, stdout='', stderr='')
            return SimpleNamespace(returncode=1, stdout='', stderr='compiler failed')

        with patch.object(builder.subprocess, 'run', side_effect=compile_archive), patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'compiler failed'):
                builder.build_toe_and_tox(self.root, executable)
        for name in ('colab-sd.toe', 'StreamDiffusion_Remote.tox'):
            self.assertEqual((self.root / name).read_bytes(), b'previous working output')

    def test_successful_build_packs_component_metadata_and_publishes_both_outputs(self):
        executable = self.make_build_sources()

        def compile_archive(command, cwd, **kwargs):
            if command[-1].endswith('.tox'):
                toc = (Path(cwd) / 'StreamDiffusion_Remote.tox.toc').read_text().splitlines()
                self.assertTrue({'.n', '.parm', '.cparm', '.build'}.issubset(toc))
                self.assertNotIn('noise1', (Path(cwd) / 'StreamDiffusion_Remote.tox.dir/.n').read_text())
                self.assertIn('noise1', (Path(cwd) / 'colab-sd.toe.dir/project1/StreamDiffusion_Remote.n').read_text())
            (Path(cwd) / command[-1]).write_bytes(b'x' * 256)
            return SimpleNamespace(returncode=0, stdout='', stderr='')

        with patch.object(builder.subprocess, 'run', side_effect=compile_archive), patch('builtins.print'):
            results = builder.build_toe_and_tox(self.root, executable)
        self.assertTrue(all(path.stat().st_size == 256 for path in results))

    def test_notebook_cells_compile_and_all_helper_sources_are_embedded(self):
        notebook = generate_notebook.make_notebook()
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
        assignments = {node.targets[0].id: ast.literal_eval(node.value)
                       for node in ast.parse(''.join(notebook['cells'][3]['source'])).body
                       if isinstance(node, ast.Assign)}
        for variable, filename in (('server_code', 'colab_server.py'), ('protocol_code', 'bridge_protocol.py'),
                                   ('launcher_code', 'colab_launcher.py')):
            self.assertEqual(assignments[variable], (generate_notebook.PROJECT_DIR / filename).read_text(encoding='utf-8'))

    def test_notebook_check_detects_source_drift(self):
        for filename in ('colab_server.py', 'colab_setup.py', 'colab_launcher.py', 'bridge_protocol.py'):
            (self.root / filename).write_text('# initial source\n')
        generate_notebook.generate_notebook(self.root)
        generate_notebook.generate_notebook(self.root, check=True)
        (self.root / 'bridge_protocol.py').write_text('# changed source\n')
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            generate_notebook.generate_notebook(self.root, check=True)

    def test_launcher_uses_readiness_response_and_fails_fast_on_process_exit(self):
        session = colab_launcher.BridgeSession(root=self.root)
        session.processes['server'] = SimpleNamespace(poll=lambda: None)
        with patch.object(colab_launcher.urllib.request, 'urlopen', return_value=io.StringIO('{"ready": true}')):
            session.wait_for_ready()
        session.processes['server'] = SimpleNamespace(poll=lambda: 1)
        with self.assertRaisesRegex(RuntimeError, 'exited with code 1'):
            session.wait_for_ready()

    def test_launcher_startup_timeout_is_bounded(self):
        session = colab_launcher.BridgeSession(root=self.root, startup_timeout=1)
        with patch.object(colab_launcher.time, 'monotonic', side_effect=[0, 2]):
            with self.assertRaises(TimeoutError):
                session.wait_for_ready()

    def test_launcher_collects_url_across_log_chunks_and_stops_owned_processes(self):
        session = colab_launcher.BridgeSession(root=self.root)
        process = Mock()
        process.poll.return_value = None
        with patch.object(colab_launcher.subprocess, 'Popen', return_value=process) as popen, patch('builtins.print'):
            session._spawn('tunnel', ['cloudflared'])
            self.assertNotEqual(popen.call_args.kwargs['stdout'], subprocess.PIPE)
            writer = session.writers['tunnel']
            writer.write('https://example-tunnel.trycloud')
            writer.flush()
            session.pump_logs()
            writer.write('flare.com\n')
            writer.flush()
            session.pump_logs()
            self.assertEqual(session.tunnel_url, 'https://example-tunnel.trycloudflare.com')
            session.stop()
        process.terminate.assert_called_once()
        self.assertFalse(session.processes)

    def test_sync_dry_run_and_check_do_not_commit_or_push(self):
        with patch.object(syncer, 'run_cmd') as run, patch.object(syncer, 'build_and_check') as build, patch('builtins.print'):
            syncer.sync('message', self.root, dry_run=True)
            run.assert_not_called()
            build.assert_not_called()
            syncer.sync('message', self.root, check_only=True)
            build.assert_called_once()
            run.assert_not_called()

    def test_sync_commands_keep_message_literal_and_do_not_use_a_shell(self):
        message = 'A "quoted" message with $(literal) and `literal`'
        with patch.object(syncer.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            syncer.run_cmd(['git', 'commit', '--message', message], self.root)
        self.assertEqual(run.call_args.args[0][-1], message)
        self.assertFalse(run.call_args.kwargs.get('shell', False))

    def test_sync_build_failure_never_stages_or_pushes(self):
        with patch.object(syncer, 'build_and_check', side_effect=syncer.SyncError('failed tests')), \
                patch.object(syncer, 'run_cmd') as run:
            with self.assertRaises(syncer.SyncError):
                syncer.sync('message', self.root)
        run.assert_not_called()

    def test_missing_pr_description_stops_before_build_or_publish(self):
        with patch.object(syncer, 'build_and_check') as build, patch.object(syncer, 'run_cmd') as run:
            with self.assertRaisesRegex(syncer.SyncError, 'does not exist'):
                syncer.sync('message', self.root, pr_body_file=self.root / 'missing.txt')
        build.assert_not_called()
        run.assert_not_called()

    def test_tooling_tests_are_not_excluded_by_gitignore(self):
        result = subprocess.run(['git', 'check-ignore', '--no-index', '--quiet', 'tests/test_project_tools.py'],
                                cwd=generate_notebook.PROJECT_DIR, capture_output=True)
        if result.returncode == 128:
            self.skipTest('A Git checkout is needed to check ignore rules')
        self.assertEqual(result.returncode, 1, 'The build-and-sync tests must be included in GitHub sync')

    def test_sync_retries_unpushed_commit_with_a_clean_working_tree(self):
        commands = []

        def run(command, root, **kwargs):
            commands.append(command)
            return SimpleNamespace(returncode=0, stdout='main' if command[1] == 'branch' else '', stderr='')

        with patch.object(syncer, 'build_and_check'), patch.object(syncer, 'run_cmd', side_effect=run), patch('builtins.print'):
            syncer.sync('message', self.root)
        self.assertTrue(any(command[:2] == ['git', 'push'] for command in commands))
        self.assertFalse(any(command[:2] == ['git', 'commit'] for command in commands))

    def test_protected_branch_creates_pr_without_automatic_admin_merge(self):
        commands = []

        def run(command, root, **kwargs):
            commands.append(command)
            if command[1] == 'branch':
                return SimpleNamespace(returncode=0, stdout='main', stderr='')
            if command[:2] == ['git', 'push'] and command[-1] == 'main':
                return SimpleNamespace(returncode=1, stdout='', stderr='GH013 protected branch')
            if command[:3] == ['gh', 'pr', 'create']:
                body = Path(command[command.index('--body-file') + 1]).read_text()
                self.assertIn('\n\nValidation:', body)
                return SimpleNamespace(returncode=0, stdout='https://github.com/example/repo/pull/1', stderr='')
            return SimpleNamespace(returncode=0, stdout='', stderr='')

        with patch.object(syncer, 'build_and_check'), patch.object(syncer, 'run_cmd', side_effect=run), patch('builtins.print'):
            url = syncer.sync('message', self.root)
        self.assertTrue(url.endswith('/1'))
        self.assertFalse(any(command[:3] == ['gh', 'pr', 'merge'] for command in commands))
        self.assertTrue(any(command[:3] == ['git', 'switch', '-c'] and command[-1].startswith('codex/') for command in commands))

    def test_protected_branch_uses_custom_pr_description_verbatim(self):
        description = 'Improve streaming reliability.\n\nValidation: project checks pass.\nLiteral `code` and $(text).\n'
        description_path = self.root / 'pr-body.txt'
        description_path.write_text(description, encoding='utf-8')

        def run(command, root, **kwargs):
            if command[1] == 'branch':
                return SimpleNamespace(returncode=0, stdout='main', stderr='')
            if command[:2] == ['git', 'push'] and command[-1] == 'main':
                return SimpleNamespace(returncode=1, stdout='', stderr='GH013 protected branch')
            if command[:3] == ['gh', 'pr', 'create']:
                self.assertEqual(Path(command[command.index('--body-file') + 1]).read_text(encoding='utf-8'), description)
                return SimpleNamespace(returncode=0, stdout='https://github.com/example/repo/pull/2', stderr='')
            return SimpleNamespace(returncode=0, stdout='', stderr='')

        with patch.object(syncer, 'build_and_check'), patch.object(syncer, 'run_cmd', side_effect=run), patch('builtins.print'):
            self.assertTrue(syncer.sync('message', self.root, pr_body_file=description_path).endswith('/2'))

    def test_network_failure_does_not_trigger_protected_branch_fallback(self):
        commands = []

        def run(command, root, **kwargs):
            commands.append(command)
            if command[1] == 'branch':
                return SimpleNamespace(returncode=0, stdout='main', stderr='')
            return SimpleNamespace(returncode=1 if command[1] == 'push' else 0, stdout='', stderr='connection reset')

        with patch.object(syncer, 'build_and_check'), patch.object(syncer, 'run_cmd', side_effect=run):
            with self.assertRaisesRegex(syncer.SyncError, 'Push failed'):
                syncer.sync('message', self.root)
        self.assertFalse(any(command[:2] == ['git', 'switch'] for command in commands))
