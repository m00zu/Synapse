"""Tests for setup_helper — pure-logic, no Qt."""
import json

import pytest

from synapse.mcp.setup_helper import (
    claude_code_command,
    claude_desktop_config_path,
    claude_desktop_entry,
    mcp_url,
    write_claude_desktop_config,
)


def test_claude_code_command_includes_port():
    cmd = claude_code_command(51780)
    assert '51780' in cmd
    assert cmd.startswith('claude mcp add synapse')
    assert '/mcp' in cmd


def test_claude_code_command_quotes_url_safely():
    cmd = claude_code_command(8765)
    # The url must appear (possibly quoted, depending on shell flavor) — just
    # verify the port is present and there's no obvious injection vector.
    assert '8765' in cmd
    assert ';' not in cmd
    assert '&' not in cmd


def test_mcp_url():
    assert mcp_url(51780) == 'http://127.0.0.1:51780/mcp'


def test_claude_desktop_entry_uses_sys_executable_by_default():
    import sys
    entry = claude_desktop_entry()
    assert entry['synapse']['command'] == sys.executable
    assert entry['synapse']['args'] == ['-m', 'synapse.mcp.bridge_stdio']


def test_claude_desktop_entry_respects_override():
    entry = claude_desktop_entry(python_path='/custom/python')
    assert entry['synapse']['command'] == '/custom/python'


def test_write_creates_file_when_missing(tmp_path):
    cfg = tmp_path / 'claude_desktop_config.json'
    result = write_claude_desktop_config(cfg, python_path='/x/python')
    assert cfg.is_file()
    data = json.loads(cfg.read_text())
    entry = data['mcpServers']['synapse']
    assert entry['command'] == '/x/python'
    assert entry['args'] == ['-m', 'synapse.mcp.bridge_stdio']
    # cwd is set so the bridge subprocess can find the synapse package.
    assert 'cwd' in entry
    assert result['config_path'] == str(cfg)
    assert result['replaced'] is False
    assert result['other_servers'] == []


def test_write_preserves_other_servers(tmp_path):
    cfg = tmp_path / 'claude_desktop_config.json'
    cfg.write_text(json.dumps({
        'mcpServers': {
            'something_else': {'command': 'other', 'args': []},
        },
        'theme': 'dark',
    }))
    result = write_claude_desktop_config(cfg, python_path='/x/python')
    data = json.loads(cfg.read_text())
    assert 'something_else' in data['mcpServers']
    assert 'synapse' in data['mcpServers']
    assert data['theme'] == 'dark'    # unrelated keys survive
    assert result['other_servers'] == ['something_else']
    assert result['replaced'] is False


def test_write_marks_replaced_when_synapse_already_present(tmp_path):
    cfg = tmp_path / 'claude_desktop_config.json'
    cfg.write_text(json.dumps({
        'mcpServers': {
            'synapse': {'command': 'old', 'args': ['old']},
        },
    }))
    result = write_claude_desktop_config(cfg, python_path='/new/python')
    data = json.loads(cfg.read_text())
    assert data['mcpServers']['synapse']['command'] == '/new/python'
    assert result['replaced'] is True


def test_write_refuses_to_clobber_malformed_json(tmp_path):
    cfg = tmp_path / 'claude_desktop_config.json'
    cfg.write_text('{this is not valid json')
    with pytest.raises(ValueError) as exc:
        write_claude_desktop_config(cfg)
    assert 'JSON' in str(exc.value)


def test_write_refuses_to_clobber_wrong_shape(tmp_path):
    cfg = tmp_path / 'claude_desktop_config.json'
    cfg.write_text(json.dumps({'mcpServers': 'not a dict'}))
    with pytest.raises(ValueError) as exc:
        write_claude_desktop_config(cfg)
    assert 'object' in str(exc.value).lower()


def test_claude_desktop_config_path_is_os_dependent():
    p = claude_desktop_config_path()
    assert p.name == 'claude_desktop_config.json'
    assert 'Claude' in p.parts
