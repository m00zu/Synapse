"""Unit tests for get_plugin_dir() routing on different install layouts."""
from pathlib import Path

import pytest

from synapse.plugin_loader import _is_macos_app_bundle, _is_versioned_extraction


class TestIsVersionedExtraction:
    def test_windows_onefile_extraction(self):
        p = Path('C:/Users/s/AppData/Local/Synapse/0.2.0/synapse.exe')
        assert _is_versioned_extraction(p)

    def test_windows_onefile_four_part_version(self):
        p = Path('C:/Users/s/AppData/Local/Synapse/1.0.0.0/synapse.exe')
        assert _is_versioned_extraction(p)

    def test_windows_standalone_install_dir(self):
        p = Path('C:/Program Files/Synapse/Synapse.exe')
        assert not _is_versioned_extraction(p)

    def test_portable_install_on_d_drive(self):
        p = Path('D:/portable/Synapse/Synapse.exe')
        assert not _is_versioned_extraction(p)

    def test_dev_mode_path(self):
        p = Path('/home/dev/PySide_Node/main.py')
        assert not _is_versioned_extraction(p)

    def test_synapse_dir_followed_by_non_version(self):
        # "plugins" is not a version string
        p = Path('C:/Users/s/AppData/Local/Synapse/plugins/foo.dll')
        assert not _is_versioned_extraction(p)


class TestIsMacosAppBundle:
    def test_inside_app_bundle(self):
        p = Path('/Applications/Synapse.app/Contents/MacOS/Synapse')
        assert _is_macos_app_bundle(p)

    def test_dev_python_on_macos(self):
        p = Path('/Users/dev/PySide_Node/main.py')
        assert not _is_macos_app_bundle(p)

    def test_plain_binary_outside_bundle(self):
        p = Path('/usr/local/bin/synapse')
        assert not _is_macos_app_bundle(p)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
