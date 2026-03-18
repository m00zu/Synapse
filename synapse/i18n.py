"""
i18n.py
=======
Minimal internationalisation helper.

Usage
-----
    from i18n import tr

    label = tr("Save Workflow")   # returns Chinese if zh_TW is active

Language preference is stored in ~/.synapse/settings.json (or the
platform-appropriate Synapse config directory in a frozen build).
"""
from __future__ import annotations
import json
import os
import platform
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_lang: str = 'en'


# ---------------------------------------------------------------------------
# Config-dir logic (mirrors plugin_loader.get_plugin_dir)
# ---------------------------------------------------------------------------
def _get_config_dir() -> Path:
    if getattr(sys, 'frozen', False):
        system = platform.system()
        if system == 'Darwin':
            base = Path.home() / 'Library' / 'Application Support' / 'Synapse'
        elif system == 'Windows':
            base = Path(os.environ.get('APPDATA', str(Path.home()))) / 'Synapse'
        else:
            base = Path.home() / '.synapse'
    else:
        base = Path.home() / '.synapse'
    base.mkdir(parents=True, exist_ok=True)
    return base


def _settings_path() -> Path:
    return _get_config_dir() / 'settings.json'


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------
def load_language() -> None:
    """Read language preference from disk and apply it. Call once at startup."""
    global _lang
    p = _settings_path()
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            _lang = data.get('language', 'en')
        except Exception:
            _lang = 'en'


def save_language(lang: str) -> None:
    """Persist language preference to disk."""
    p = _settings_path()
    data: dict = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            pass
    data['language'] = lang
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def get_language() -> str:
    return _lang


def set_language(lang: str) -> None:
    global _lang
    _lang = lang
    save_language(lang)


# ---------------------------------------------------------------------------
# Translation function
# ---------------------------------------------------------------------------
def tr(s: str) -> str:
    """Return the translation of *s* in the current language, or *s* itself."""
    if _lang == 'en':
        return s
    from .translations.zh_TW import STRINGS as _zh
    table = {'zh_TW': _zh}.get(_lang, {})
    return table.get(s, s)
