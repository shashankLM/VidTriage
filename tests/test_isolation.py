"""The suite must not write into the developer's real config directory.

This is not hypothetical. ``CONFIG_DIR`` is computed once from ``Path.home()``
at import time, and the session-scoped fixture that used to redirect ``HOME``
ran too late — test modules had already imported the constant during collection.
The suite spent its life writing real sessions and triage logs into
``~/.vidtriage``, and because the session list is most-recently-used first, the
app then restored a pytest temp directory on the next real launch.

If this file fails, every other test is silently touching your home directory.
"""

from __future__ import annotations

from conftest import ISOLATED_HOME

from vidtriage.persistence.settings import CONFIG_DIR, default_settings_path, user_plugin_dir
from vidtriage.plugins.builtin.triage.config import SESSIONS_FILE
from vidtriage.plugins.builtin.triage.ledger import default_log_dir


def test_config_dir_is_redirected():
    assert CONFIG_DIR.is_relative_to(ISOLATED_HOME), (
        f"CONFIG_DIR is {CONFIG_DIR}, which is outside the test home — "
        f"the suite is reading and writing the real ~/.vidtriage."
    )


def test_every_writable_location_is_inside_the_test_home(tmp_path):
    """One escapee is enough to corrupt a real session list."""
    for path in (
        default_settings_path(),
        user_plugin_dir(),
        SESSIONS_FILE,
        default_log_dir(tmp_path),
    ):
        assert path.is_relative_to(ISOLATED_HOME), f"{path} escapes the test home"
