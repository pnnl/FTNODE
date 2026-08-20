"""Execute notebooks in place so they carry stored outputs.

Usage::

    uv run python scripts/execute_notebooks.py examples/duffing/pkg_kappa_variants.ipynb
    uv run python scripts/execute_notebooks.py --kernel ftnode <nb> [<nb> ...]

Register the kernel first, once, so it points at this project's interpreter by
absolute path::

    uv run python -m ipykernel install --user --name ftnode --display-name "FTNODE (uv)"

The ``--name`` matters: without it the spec is written as ``python3`` and
clobbers whatever global ``python3`` kernel you already had.  Note also that the
kernelspec uv drops in ``.venv/share/jupyter/kernels/python3`` uses a bare
``python`` in its ``argv``, which resolves off ``PATH`` at launch and can start a
completely different interpreter -- that is the failure this script's ``--kernel``
default sidesteps.

The notebook's own ``metadata.kernelspec`` is left alone: it stays the portable
``python3`` so the file opens sensibly for someone who has not registered a
project kernel.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import nbformat
from nbclient import NotebookClient


def execute(path: pathlib.Path, kernel: str, timeout: int) -> None:
    nb = nbformat.read(path, as_version=4)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"[{path.name}] executing {n_code} code cells with kernel {kernel!r} ...", flush=True)

    t0 = time.time()
    client = NotebookClient(
        nb,
        kernel_name=kernel,
        timeout=timeout,
        # Run with the notebook's own directory as cwd: these notebooks load and
        # write checkpoints by bare relative filename.
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    nbformat.write(nb, path)
    print(f"[{path.name}] done in {time.time() - t0:.0f}s -> outputs written", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("notebooks", nargs="+", type=pathlib.Path)
    ap.add_argument("--kernel", default="ftnode", help="kernelspec name (default: ftnode)")
    ap.add_argument("--timeout", type=int, default=3600, help="per-cell timeout in seconds")
    args = ap.parse_args(argv)

    for p in args.notebooks:
        if not p.exists():
            print(f"error: {p} not found", file=sys.stderr)
            return 1
        execute(p, args.kernel, args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
