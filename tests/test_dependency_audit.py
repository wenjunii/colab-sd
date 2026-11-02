from types import SimpleNamespace
import unittest

from audit_dependencies import installed_requirements, validate_report


class DependencyAuditChecks(unittest.TestCase):
    def test_official_cpu_build_and_all_transitive_packages_are_audited(self):
        installed = [SimpleNamespace(metadata={'Name': 'torch'}, version='2.13.0+cpu'),
                     SimpleNamespace(metadata={'Name': 'HuggingFace_Hub'}, version='1.33.0')]
        expected = installed_requirements(installed)
        self.assertEqual(expected, {'torch': '2.13.0', 'huggingface-hub': '1.33.0'})
        report = {'dependencies': [{'name': name, 'version': version, 'vulns': []}
                                   for name, version in expected.items()]}
        validate_report(report, expected)

    def test_other_local_builds_and_empty_inventories_do_not_pass(self):
        for name, version in [('torch', '2.13.0+cu130'), ('example', '1.0+custom')]:
            with self.assertRaisesRegex(ValueError, 'unsupported local build'):
                installed_requirements([SimpleNamespace(metadata={'Name': name}, version=version)])
        with self.assertRaisesRegex(ValueError, 'No installed packages'):
            installed_requirements([])

    def test_skipped_missing_or_wrong_version_results_do_not_pass(self):
        for dependencies in [[], [{'name': 'torch', 'skip_reason': 'not found'}],
                             [{'name': 'torch', 'version': '2.5.1', 'vulns': []}]]:
            with self.subTest(dependencies=dependencies), self.assertRaises(ValueError):
                validate_report({'dependencies': dependencies}, {'torch': '2.13.0'})

    def test_a_vulnerability_never_passes_even_with_a_complete_inventory(self):
        report = {'dependencies': [{'name': 'torch', 'version': '2.5.1',
                                    'vulns': [{'id': 'GHSA-53q9-r3pm-6pq6'}]}]}
        with self.assertRaisesRegex(ValueError, 'GHSA-53q9-r3pm-6pq6'):
            validate_report(report, {'torch': '2.5.1'})
