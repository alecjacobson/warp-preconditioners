"""Cache the Laplacian, invalidating all derived caches when inputs change."""

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as ss


def load_laplacian(path=Path("data/dirichlet")):
    source = Path("/tmp/dump/k4_Q.mtx")
    stat = source.stat()
    data = np.load(path / "fields.npz")
    digest = hashlib.sha256()
    for key in ("vertices", "mass", "free", "constraints"):
        digest.update(data[key].tobytes())
    signature = dict(
        source=str(source.resolve()),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        data_sha256=digest.hexdigest(),
    )
    stamp = path / "tuning_inputs.json"
    cache = path / "tuning_laplacian.npz"
    if not (stamp.exists() and cache.exists() and json.loads(stamp.read_text()) == signature):
        mixed = sio.mmread(source).tocsr()
        n = len(data["vertices"])
        laplacian = -mixed[:n, n:]
        ss.save_npz(cache, laplacian)
        for name in ("tuning_matrix.npz", "tuning_rhs.npy"):
            (path / name).unlink(missing_ok=True)
        stamp.write_text(json.dumps(signature, indent=2) + "\n")
        return laplacian
    return ss.load_npz(cache)
