# Restore the complete research snapshot

A Git clone contains code, reports and papers; it does not contain the complete data/models.

Download every asset from release `research-snapshot-20261107T032034Z` into one folder, including `restore.py`, `inventory.json`, `snapshot.json`, `SHA256SUMS` and all `snapshot.tar.gz.partNNN` files. Then run:

```bash
python3 restore.py --snapshot-dir . --destination /path/to/empty-restore-folder
```

The destination must not already exist. The script checks checksums, safely extracts the archive, and verifies every restored file and symlink. The restored project includes the original `.git` directory, including local history and the original uncommitted state.

The snapshot includes all files present in the original project, including ignored data, models and caches. Third-party datasets and code retain their original terms; this backup does not grant a new license. Preserve attribution and consult recorded source provenance before redistributing or reusing third-party material. Files referenced outside the project are not included. A referenced Temporal_Walk_Counterfactual_Audits directory was absent at preparation time; this snapshot cannot recover that missing directory.
