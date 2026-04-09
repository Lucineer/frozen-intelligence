#!/usr/bin/env python3
"""Weight image version control for mask-locked chips.

Track weight changes across versions, diff between weight sets,
compute similarity scores, and manage weight branches.
"""
import hashlib, struct, time, math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class WeightLayer:
    name: str
    precision: str
    rows: int
    cols: int
    data: bytes  # raw weight data

    @property
    def size(self) -> int:
        return len(self.data)

    def hash(self) -> str:
        return hashlib.sha256(self.data).hexdigest()[:16]

    def mean(self) -> float:
        if not self.data:
            return 0.0
        return sum(b for b in self.data) / len(self.data)

    def norm(self) -> float:
        return math.sqrt(sum(b * b for b in self.data)) if self.data else 0.0


@dataclass
class WeightCommit:
    version: str
    timestamp: float
    message: str
    parent: Optional[str] = None
    layers: Dict[str, WeightLayer] = field(default_factory=dict)

    def hash(self) -> str:
        content = f"{self.version}:{self.message}:{self.parent or ''}"
        for name, layer in sorted(self.layers.items()):
            content += f":{name}:{layer.hash()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]


class WeightVCS:
    """Version control system for weight images."""

    def __init__(self, repo_name: str = "weights"):
        self.repo_name = repo_name
        self.commits: Dict[str, WeightCommit] = {}
        self.branches: Dict[str, str] = {"main": None}
        self.head = "main"

    def commit(self, message: str, layers: Dict[str, WeightLayer],
               version: str = None) -> str:
        """Create a new commit."""
        if version is None:
            version = f"v{len(self.commits) + 1}"

        parent = self.branches.get(self.head)
        ts = time.time()

        commit = WeightCommit(version, ts, message, parent, dict(layers))
        commit_hash = commit.hash()
        self.commits[commit_hash] = commit
        self.branches[self.head] = commit_hash

        return commit_hash

    def diff(self, hash_a: str, hash_b: str) -> Dict:
        """Compare two weight commits."""
        ca = self.commits.get(hash_a)
        cb = self.commits.get(hash_b)
        if not ca or not cb:
            return {"error": "commit not found"}

        result = {"layers_changed": [], "layers_added": [], "layers_removed": [],
                 "total_bytes_changed": 0}

        # Check all layers
        all_names = set(list(ca.layers.keys()) + list(cb.layers.keys()))

        for name in all_names:
            in_a = name in ca.layers
            in_b = name in cb.layers

            if in_a and not in_b:
                result["layers_removed"].append(name)
            elif in_b and not in_a:
                result["layers_added"].append(name)
            else:
                la = ca.layers[name]
                lb = cb.layers[name]
                if la.hash() != lb.hash():
                    bytes_diff = sum(1 for a, b in zip(la.data, lb.data) if a != b)
                    total = max(len(la.data), len(lb.data))
                    pct = bytes_diff / total * 100 if total > 0 else 0
                    result["layers_changed"].append({
                        "name": name,
                        "bytes_different": bytes_diff,
                        "total_bytes": total,
                        "change_pct": round(pct, 1),
                        "hash_before": la.hash(),
                        "hash_after": lb.hash(),
                    })
                    result["total_bytes_changed"] += bytes_diff

        return result

    def log(self, branch: str = None, limit: int = 10) -> List[Dict]:
        """Show commit history."""
        head = self.branches.get(branch or self.head)
        if not head:
            return []

        history = []
        current = head
        visited = set()
        while current and current not in visited and len(history) < limit:
            visited.add(current)
            commit = self.commits[current]
            history.append({
                "version": commit.version,
                "hash": current,
                "message": commit.message,
                "parent": commit.parent,
                "layers": len(commit.layers),
                "timestamp": time.strftime("%Y-%m-%d %H:%M",
                            time.localtime(commit.timestamp)),
            })
            current = commit.parent
        return history

    def branch(self, name: str, from_commit: str = None):
        """Create a new branch."""
        source = from_commit or self.branches.get(self.head)
        self.branches[name] = source

    def checkout(self, branch: str):
        """Switch to a branch."""
        if branch in self.branches:
            self.head = branch

    def similarity(self, hash_a: str, hash_b: str) -> Dict:
        """Compute weight similarity between commits."""
        ca = self.commits.get(hash_a)
        cb = self.commits.get(hash_b)
        if not ca or not cb:
            return {"error": "commit not found"}

        common = set(ca.layers.keys()) & set(cb.layers.keys())
        results = {}

        total_similarity = 0
        count = 0

        for name in common:
            la = ca.layers[name]
            lb = cb.layers[name]
            min_len = min(len(la.data), len(lb.data))
            if min_len == 0:
                continue

            # Hamming distance
            matches = sum(1 for i in range(min_len) if la.data[i] == lb.data[i])
            similarity = matches / min_len

            # Cosine similarity (normalized)
            dot = sum(a * b for a, b in zip(la.data, lb.data))
            norm_a = la.norm() or 1
            norm_b = lb.norm() or 1
            cosine = dot / (norm_a * norm_b)

            results[name] = {"hamming": round(similarity, 4), "cosine": round(cosine, 4)}
            total_similarity += similarity
            count += 1

        avg = total_similarity / count if count > 0 else 0
        return {"per_layer": results, "average_similarity": round(avg, 4),
                "layers_compared": count}


def demo():
    print("=== Weight Image Version Control ===\n")

    vcs = WeightVCS("frozen_scout")

    # Generate weight data
    def make_layers(seed: int, n_layers: int = 3):
        import random
        random.seed(seed)
        layers = {}
        for i in range(n_layers):
            name = f"layer_{i}"
            data = bytes(random.randint(0, 255) for _ in range(100))
            layers[name] = WeightLayer(name, "INT4", 10, 10, data)
        return layers

    # Commit history
    c1 = vcs.commit("Initial weights", make_layers(42))
    c2 = vcs.commit("Fine-tuned on domain data", make_layers(43))
    c3 = vcs.commit("Quantized to INT4", make_layers(44))

    print("--- Commit Log ---")
    for entry in vcs.log():
        print(f"  {entry['hash']} {entry['version']} {entry['timestamp']}: {entry['message']} ({entry['layers']} layers)")

    # Diff
    print("\n--- Diff (c1 vs c3) ---")
    diff = vcs.diff(c1, c3)
    print(f"  Changed: {len(diff['layers_changed'])}")
    for layer in diff["layers_changed"]:
        print(f"    {layer['name']}: {layer['change_pct']:.1f}% changed ({layer['bytes_different']}/{layer['total_bytes']} bytes)")
    print(f"  Total bytes changed: {diff['total_bytes_changed']}")

    # Similarity
    print("\n--- Similarity (c1 vs c2) ---")
    sim = vcs.similarity(c1, c2)
    print(f"  Average: {sim['average_similarity']:.2%}")
    for name, s in sim["per_layer"].items():
        print(f"    {name}: hamming={s['hamming']:.4f}, cosine={s['cosine']:.4f}")

    # Branching
    print("\n--- Branches ---")
    vcs.branch("experimental", c1)
    c4 = vcs.commit("Experimental architecture", make_layers(99))
    print(f"  main: {vcs.branches['main']}")
    print(f"  experimental: {vcs.branches['experimental']}")

    vcs.checkout("experimental")
    c5 = vcs.commit("MoE routing v2", make_layers(100))
    print(f"  experimental head: {vcs.branches['experimental']}")
    print(f"  Experimental log:")
    for entry in vcs.log():
        print(f"    {entry['hash']} {entry['message']}")


if __name__ == "__main__":
    demo()
