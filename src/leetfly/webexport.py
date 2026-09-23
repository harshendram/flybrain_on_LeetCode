"""Helpers for shipping arrays to the browser (web/src/fly.ts decodes them with `decodeArrays`)."""

import numpy as np


class Blob:
    """Concatenates typed arrays into one binary; the manifest records (offset, length, dtype) for each."""

    def __init__(self):
        self.parts, self.manifest, self.offset = [], {}, 0

    def add(self, name: str, arr: np.ndarray, dtype: str) -> None:
        a = np.ascontiguousarray(arr, dtype=dtype)
        pad = (-self.offset) % 8
        if pad:
            self.parts.append(b"\0" * pad)
            self.offset += pad
        self.manifest[name] = {"offset": self.offset, "length": int(a.size), "dtype": dtype, "shape": list(a.shape)}
        self.parts.append(a.tobytes())
        self.offset += a.nbytes

    def write(self, path) -> None:
        path.write_bytes(b"".join(self.parts))
