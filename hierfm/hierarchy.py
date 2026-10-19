"""Hierarchy as a tree of named nodes and its summing matrix S (all nodes x bottom nodes).

hierarchy.json:
    {"levels": ["UES", "IPS", "region"],
     "nodes": [{"id": "UES", "parent": null, "attrs": {...}},
               {"id": "IPS_CENTER", "parent": "UES", "attrs": {"tz": 3}}, ...]}

Node order in S is top-down, level by level, so slices by level stay contiguous.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Hierarchy:
    nodes: list            # all node ids, top-down
    bottom: list           # leaf ids, in S column order
    level: dict            # node id -> level index
    levels: list           # level names
    parent: dict           # node id -> parent id or None
    attrs: dict            # node id -> free attributes (ips, tz, lat, lon, ...)
    S: np.ndarray          # (n, m) summing matrix

    @property
    def n(self):
        return len(self.nodes)

    @property
    def m(self):
        return len(self.bottom)

    def level_index(self, lv):
        return [i for i, v in enumerate(self.nodes) if self.level[v] == lv]

    def ancestor_at(self, node, lv):
        while node is not None and self.level[node] > lv:
            node = self.parent[node]
        return node

    def aggregate(self, bottom_values):
        """(..., m) bottom values -> (..., n) values for every node."""
        return bottom_values @ self.S.T


def from_nodes(levels, records):
    parent = {r["id"]: r.get("parent") for r in records}
    attrs = {r["id"]: r.get("attrs", {}) for r in records}
    children = {k: [] for k in parent}
    for k, p in parent.items():
        if p is not None:
            children[p].append(k)

    level = {}
    roots = [k for k, p in parent.items() if p is None]
    frontier, depth = roots, 0
    while frontier:
        for k in frontier:
            level[k] = depth
        frontier = [c for k in frontier for c in children[k]]
        depth += 1
    nodes = sorted(parent, key=lambda k: (level[k], [r["id"] for r in records].index(k)))
    bottom = [k for k in nodes if not children[k]]

    col = {b: j for j, b in enumerate(bottom)}
    S = np.zeros((len(nodes), len(bottom)))
    for i, k in enumerate(nodes):
        stack = [k]
        while stack:
            x = stack.pop()
            if children[x]:
                stack.extend(children[x])
            else:
                S[i, col[x]] = 1.0
    return Hierarchy(nodes, bottom, level, list(levels), parent, attrs, S)


def load(path):
    raw = json.loads(Path(path).read_text())
    return from_nodes(raw["levels"], raw["nodes"])


def save(h, path):
    recs = [{"id": k, "parent": h.parent[k], "attrs": h.attrs[k]} for k in h.nodes]
    Path(path).write_text(json.dumps({"levels": h.levels, "nodes": recs}, ensure_ascii=False, indent=1))
