"""Guard: every raw-binary read/write in neurobox pins its byte order.

neurosuite-3 files are little-endian on disk regardless of the host, so
every ``np.fromfile`` / ``np.frombuffer`` / ``np.memmap`` call must name
an explicit byte order (``"<i8"``, ``np.dtype("<f4")``, ...) rather than
a native-order alias (``np.int16``, ``"i8"``, ``int``).

This is a *source-level* check rather than a behavioural one on purpose:
on little-endian hardware ``np.int16`` and ``"<i2"`` are byte-for-byte
identical, so no runtime assertion executed on x86 or Apple Silicon can
tell them apart.  The failure mode of a native dtype is silent data
corruption on a big-endian host, which is exactly the kind of bug that
never surfaces in CI.  Checking the source is the only thing that
actually catches it.

Regression origin: ``sync_pipelines`` read the spots ``.pos`` file with
a native ``np.int16`` while all twelve other binary readers pinned
little-endian.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


# Package roots that perform raw binary I/O.
_SCAN_ROOTS = ("io", "dtype", "analysis", "viz", "config", "utils")

# Calls whose dtype argument must be explicitly byte-ordered.
_BINARY_READERS = {"fromfile", "frombuffer", "memmap"}

# A dtype string is explicit if it names an endianness, or is a
# single-byte / byte-order-irrelevant type (i1, u1, S, V, bool).
_EXPLICIT_RE = re.compile(r"^[<>|]")
_ORDER_FREE_RE = re.compile(r"^(\|?[iu]1|\|?b1|[SaV]\d*|\?)$")


def _package_root() -> Path:
    return Path(__file__).resolve().parent.parent / "neurobox"


def _python_files() -> list[Path]:
    root = _package_root()
    files: list[Path] = []
    for sub in _SCAN_ROOTS:
        d = root / sub
        if d.is_dir():
            files.extend(p for p in d.rglob("*.py") if "~" not in p.name)
    return files


def _dtype_is_explicit(node: ast.AST, module_src: str) -> tuple[bool, str]:
    """Return (ok, rendered) for a dtype argument AST node."""
    rendered = ast.get_source_segment(module_src, node) or ast.dump(node)

    # Literal string: "<i8", "i8", ...
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        s = node.value
        ok = bool(_EXPLICIT_RE.match(s) or _ORDER_FREE_RE.match(s))
        return ok, rendered

    # np.dtype("<i4") / np.dtype(f"<i{n}") — recurse into the argument
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "dtype"
            and node.args):
        return _dtype_is_explicit(node.args[0], module_src)

    # f-string: explicit iff it starts with a literal <, > or |
    if isinstance(node, ast.JoinedStr):
        first = node.values[0] if node.values else None
        ok = (isinstance(first, ast.Constant)
              and isinstance(first.value, str)
              and bool(_EXPLICIT_RE.match(first.value)))
        return ok, rendered

    # A bare Name is a module-level constant (e.g. dtype_raw,
    # _HEADER_DTYPE).  Resolved separately below.
    if isinstance(node, ast.Name):
        return True, rendered

    # np.int16, np.float64, int, ... — native order, not acceptable.
    return False, rendered


def _collect_offenders() -> list[str]:
    offenders: list[str] = []
    for path in _python_files():
        src = path.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError:                     # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (
                fn.id if isinstance(fn, ast.Name) else None)
            if name not in _BINARY_READERS:
                continue
            dtype_node = next(
                (kw.value for kw in node.keywords if kw.arg == "dtype"), None)
            if dtype_node is None:
                # positional dtype for fromfile/frombuffer is arg 1
                dtype_node = node.args[1] if len(node.args) > 1 else None
            if dtype_node is None:
                offenders.append(
                    f"{path.relative_to(_package_root().parent)}:"
                    f"{node.lineno}: np.{name}(...) with no dtype "
                    f"(defaults to native float64)")
                continue
            ok, rendered = _dtype_is_explicit(dtype_node, src)
            if not ok:
                offenders.append(
                    f"{path.relative_to(_package_root().parent)}:"
                    f"{node.lineno}: np.{name}(dtype={rendered}) is "
                    f"native byte order")
    return offenders


class TestBinaryByteOrder:
    def test_all_binary_reads_pin_byte_order(self):
        offenders = _collect_offenders()
        assert not offenders, (
            "Raw binary I/O must name an explicit byte order "
            "(e.g. \"<i2\" not np.int16):\n  "
            + "\n  ".join(offenders)
        )

    def test_named_dtype_constants_are_explicit(self):
        """Readers that hoist their dtype into a module constant
        (``dtype_raw``, ``_HEADER_DTYPE``) still have to pin order —
        the AST walk above waves those through, so check them here."""
        bad: list[str] = []
        pat = re.compile(
            r"^\s*(_?[A-Za-z_]*(?:dtype|DTYPE)[A-Za-z_]*)\s*=\s*"
            r"np\.dtype\((.+?)\)\s*$", re.MULTILINE)
        for path in _python_files():
            for m in pat.finditer(path.read_text()):
                name, arg = m.group(1), m.group(2).strip()
                inner = arg.strip("fr\"'")
                if not _EXPLICIT_RE.match(inner) and not _ORDER_FREE_RE.match(inner):
                    bad.append(
                        f"{path.relative_to(_package_root().parent)}: "
                        f"{name} = np.dtype({arg}) is native byte order")
        assert not bad, "\n  ".join(["Native-order dtype constants:"] + bad)

    def test_guard_catches_a_native_dtype(self, tmp_path):
        """Sanity-check the guard itself: it must reject np.int16."""
        src = "import numpy as np\nx = np.fromfile('f', dtype=np.int16)\n"
        tree = ast.parse(src)
        call = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.Call)
                    and getattr(n.func, "attr", None) == "fromfile")
        dtype_node = next(kw.value for kw in call.keywords if kw.arg == "dtype")
        ok, rendered = _dtype_is_explicit(dtype_node, src)
        assert not ok and "np.int16" in rendered

    @pytest.mark.parametrize("spec", ["<i2", ">f8", "|i1", "u1"])
    def test_guard_accepts_explicit_specs(self, spec):
        src = f"import numpy as np\nx = np.fromfile('f', dtype='{spec}')\n"
        tree = ast.parse(src)
        call = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.Call)
                    and getattr(n.func, "attr", None) == "fromfile")
        dtype_node = next(kw.value for kw in call.keywords if kw.arg == "dtype")
        ok, _ = _dtype_is_explicit(dtype_node, src)
        assert ok
