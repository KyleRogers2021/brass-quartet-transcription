r"""
Check every downloaded CocoChorales file against the official MD5 checksum list.

An MD5 checksum is a fingerprint of a file. If even one byte got corrupted
during download, the fingerprint won't match the official one.

Run from the Miniforge Prompt:
    python R:\code\verify_cocochorales.py
"""
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(r"R:\datasets\cocochorales\zipped")
MD5_FILE = ROOT / "cocochorales_md5s.txt"


def key(path_text):
    """Reduce a path to 'part/split/file', e.g. 'main_dataset/train/25.tar.bz2'."""
    parts = path_text.replace("\\", "/").strip("()").split("/")
    return "/".join(parts[-3:])


# 1. Read the official checksum list.
expected = {}
for line in MD5_FILE.read_text().splitlines():
    tokens = [t.strip("()") for t in line.split()]
    md5 = next((t for t in tokens if re.fullmatch(r"[0-9a-fA-F]{32}", t)), None)
    path = next((t for t in tokens if t.endswith(".tar.bz2")), None)
    if md5 and path:
        expected[key(path)] = md5.lower()

if not expected:
    print("Could not read the checksum list. First lines of the file:")
    print("\n".join(MD5_FILE.read_text().splitlines()[:5]))
    sys.exit(1)

# 2. Fingerprint each downloaded file and compare.
files = sorted(ROOT.rglob("*.tar.bz2"))
good, bad, not_listed = [], [], []
for i, f in enumerate(files, 1):
    k = key(str(f.relative_to(ROOT)))
    if k not in expected:
        not_listed.append(k)
        print(f"[{i}/{len(files)}] ??? {k} (not in checksum list)")
        continue
    h = hashlib.md5()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    ok = h.hexdigest() == expected[k]
    (good if ok else bad).append(k)
    print(f"[{i}/{len(files)}] {'OK ' if ok else 'BAD'} {k}")

# 3. Summary.
print()
print(f"Matched:   {len(good)}")
print(f"Corrupted: {len(bad)}")
print(f"Unlisted:  {len(not_listed)}")
for k in bad:
    print("  re-download:", k)
if not_listed:
    print("Example checksum-list entries:", list(expected)[:3])
