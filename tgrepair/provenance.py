"""Small source snapshots for reproducible research runs across edits."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile

from .data import sha256


def snapshot_sources(root):
    root=Path(root)
    factory=root.parent/'RF_Dataset_Factory'
    files={}
    for folder in ['tgrepair','scripts','tests']:
        for p in sorted((root/folder).glob('*.py')):
            files['research/'+str(p.relative_to(root))]=p
    for p in sorted((factory/'src/temporal_factory').glob('*.py')):
        files['factory/'+str(p.relative_to(factory))]=p
    for name in ['src/rfdf/__init__.py','src/rfdf/storage.py','pyproject.toml']:
        files['factory/'+name]=factory/name
    contents={name:path.read_bytes() for name,path in files.items()}
    hashes={name:hashlib.sha256(content).hexdigest() for name,content in contents.items()}
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    dest=root/'data/source_snapshots'/f'{digest}.tar.gz'
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode='wb',dir=dest.parent,prefix=digest+'.',suffix='.tmp',delete=False) as raw:
                temporary=Path(raw.name)
                with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz:
                    with tarfile.open(fileobj=gz,mode='w|') as archive:
                        for name,content in contents.items():
                            info=tarfile.TarInfo(name);info.size=len(content);info.mtime=0;info.mode=0o644
                            archive.addfile(info,io.BytesIO(content))
            # Same-content concurrent writers produce identical complete archives.
            # The public destination never exposes a partially written gzip file.
            os.replace(temporary,dest)
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)
    return {'file':str(dest.relative_to(root)),'sha256':sha256(dest),'source_set_hash':digest,'files':hashes}
