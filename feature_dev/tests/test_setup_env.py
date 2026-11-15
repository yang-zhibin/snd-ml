"""Safety checks for the explicit lxplus environment installer."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup_env


class SetupTests(unittest.TestCase):
    def test_install_environment_is_isolated(self):
        original = {'PYTHONPATH': '/external/packages', 'PYTHONHOME': '/external/python',
                    'PIP_TARGET': '/external/target', 'PIP_PREFIX': '/external/prefix',
                    'PIP_USER': '1', 'PATH': '/usr/bin'}
        with patch.dict(os.environ, original, clear=True):
            env = setup_env.isolated_environment()
            for key in original:
                if key != 'PATH':
                    self.assertNotIn(key, env)
            self.assertEqual(env['PYTHONNOUSERSITE'], '1')
            self.assertEqual(env['PIP_CONFIG_FILE'], os.devnull)
            self.assertEqual(os.environ['PYTHONPATH'], '/external/packages')

    def test_atomic_configuration_is_plain_data(self):
        with tempfile.TemporaryDirectory() as d:
            config = Path(d) / 'config'
            value = '/directory with spaces/$(not-a-command)/bin/snakemake\n'
            setup_env.atomic_write(config, value)
            self.assertEqual(config.read_text(), value)
            self.assertEqual(list(Path(d).iterdir()), [config])

    def test_unmanaged_environment_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d) / 'existing';prefix.mkdir()
            sentinel = prefix / 'keep';sentinel.write_text('untouched')
            config = Path(d) / 'config'
            with patch.object(setup_env, 'CONFIG', config), patch.object(sys, 'argv', ['setup', '--prefix', str(prefix)]), patch.object(subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'unmanaged/different'):
                    setup_env.main()
                run.assert_not_called()
            self.assertEqual(sentinel.read_text(), 'untouched')
            self.assertFalse(config.exists())

    def test_failed_install_preserves_previous_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            config = Path(d) / 'config';config.write_text('/old/bin/snakemake\n')
            prefix = Path(d) / 'new'
            with patch.object(setup_env, 'CONFIG', config), patch.object(sys, 'argv', ['setup', '--prefix', str(prefix)]), patch.object(subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'venv')):
                with self.assertRaises(subprocess.CalledProcessError):
                    setup_env.main()
            self.assertEqual(config.read_text(), '/old/bin/snakemake\n')
            self.assertEqual(json.loads((prefix/setup_env.MARKER).read_text())['owner'], 'snd-ml-feature-dev')

    def test_check_does_not_write_config(self):
        with tempfile.TemporaryDirectory() as d:
            config = Path(d) / 'config'
            prefix = Path(d) / 'env'
            with patch.object(setup_env, 'CONFIG', config), patch.object(sys, 'argv', ['setup', '--check', '--prefix', str(prefix)]), patch.object(setup_env, 'verify', return_value=prefix/'bin/snakemake'):
                setup_env.main()
            self.assertFalse(config.exists())
            self.assertFalse(prefix.exists())


if __name__ == '__main__':
    unittest.main()
