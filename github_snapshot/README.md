# Complete research snapshot

Follow [RESTORE.md](RESTORE.md) to recover all original project files. The standalone `restore.py` uses only the Python standard library.

`prepare.py` and `publish.py` preserve the packaging implementation. They expect a separate backup workspace, rather than execution from this directory. To reuse them, place the original project folders beside a `github_temporal_backup/` directory and copy these scripts plus `restore.py` into that directory. Install GitHub CLI at `github_temporal_backup/tools/gh`; authenticate with `GH_CONFIG_DIR` set to `github_temporal_backup/.gh`. Preparation also requires Git, GNU tar and pigz. Run preparation from an unused workspace: existing snapshot and repository directories are deliberately not overwritten. Run publication after preparation completes. Credentials are local only and must never be uploaded.

The full archive preserves the original folder and Git metadata. The repository adds snapshot navigation and packaging records for public access. Scientific outputs were preserved without rerunning experiments. Third-party sources retain their own terms.
