"""Report stud-connector coverage over the parts used by real sample models.

Usage: python -m lego_tf.connector_coverage data/samples/*.mpd
"""

from __future__ import annotations

import os
import sys
from collections import Counter

from lego_tf.data.ldraw import flatten_file
from lego_tf.data.parts import PartLibrary, ConnectorExtractor, ConnType

LDRAW_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "ldraw_lib", "ldraw")


def main(paths):
    lib = PartLibrary(LDRAW_ROOT)
    ext = ConnectorExtractor(lib)

    part_counts = Counter()
    for p in paths:
        for b in flatten_file(p):
            part_counts[b.part.replace("\\", "/").split("/")[-1]] += 1

    missing, no_conn, with_conn = [], [], []
    total_studs = 0
    for part, n in part_counts.items():
        path, cat = lib.resolve(part)
        if path is None:
            missing.append((part, n))
            continue
        conns = ext.connectors(part)
        studs = sum(1 for c in conns if c.type == ConnType.STUD)
        anti = sum(1 for c in conns if c.type == ConnType.ANTISTUD)
        total_studs += studs * n
        if not conns:
            no_conn.append((part, n))
        else:
            with_conn.append((part, n, studs, anti))

    n_unique = len(part_counts)
    n_inst = sum(part_counts.values())
    inst_with = sum(n for _, n, _, _ in with_conn)
    print(f"unique parts: {n_unique} | instances: {n_inst}")
    print(f"resolved-with-connectors: {len(with_conn)} unique "
          f"({100*inst_with/n_inst:.1f}% of instances)")
    print(f"resolved-no-connectors:   {len(no_conn)} unique "
          f"({100*sum(n for _,n in no_conn)/n_inst:.1f}% of instances)")
    print(f"unresolved (missing):     {len(missing)} unique "
          f"({100*sum(n for _,n in missing)/n_inst:.1f}% of instances)")
    print(f"total male studs across all instances: {total_studs}")
    if no_conn:
        print("\ntop no-connector parts (expected for tiles/technic-beams/wheels):")
        for part, n in sorted(no_conn, key=lambda x: -x[1])[:12]:
            print(f"  {part:16s} x{n}")
    if missing:
        print("\nmissing parts:")
        for part, n in sorted(missing, key=lambda x: -x[1])[:12]:
            print(f"  {part:16s} x{n}")


if __name__ == "__main__":
    main(sys.argv[1:])
