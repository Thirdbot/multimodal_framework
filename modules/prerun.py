"""
prerun.py – Initial workspace setup.

Call create_config_folders() once before any other module is used to ensure
the config directory and saved-config file exist on disk.
"""

from modules.variable import Variable


def create_config_folders():
    """Create the configs directory and the empty saved_config.json if missing."""
    v = Variable()

    # Ensure the configs/ directory exists
    v.DMConfig_DIR.mkdir(parents=True, exist_ok=True)

    # Touch saved_config.json so other modules can open it safely
    v.SAVED_CONFIG_Path.touch(exist_ok=True)

    print(f"Config folder ready: {v.DMConfig_DIR}")


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    create_config_folders()
