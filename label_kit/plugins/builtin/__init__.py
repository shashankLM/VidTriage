"""First-party plugins.

Every subpackage here is discovered automatically at startup and must expose a
module-level ``PLUGIN`` attribute naming a :class:`~label_kit.plugins.api.Plugin`
subclass. They are ordinary plugins with no privileges beyond being shipped in
the box — which is the point: if the built-in triage workflow can be expressed
through the plugin API, so can anything you want to add.
"""
