#!/usr/bin/env python3
"""Who changed which file since the last sync.

Every Yjs item carries the clientID of the doc instance that created it, and
the document's state vector records, per clientID, how far that client has
written (its clock). Comparing a document's state vector with the one seen at
its previous successful export therefore yields exactly the clients that wrote
in between.

The Lens relay maps clientIDs to actors in the doc's top-level "users" map
(Yjs PermanentUserData layout: users.<actor> = {ids: [clientID, ...], ...}):
  - ``human:<display name>`` - Lens Editor users, registered by the editor on
    their first edit. The name is self-reported, not verified.
  - ``ai:<model>:<behalf>`` - edits made through the relay MCP.
  - a bare relay user id - Obsidian/Relay.md connections, registered by the
    server under the token's user id (no display name available here).
ClientIDs with no entry (the server's own writes, e.g. link indexing) are
ignored.

Flow: RelayClient calls ``observe`` with each fetched Y.Doc; the sync engine
calls ``confirm`` once that doc's export reached disk (or ``advance`` when the
export turned out to be a no-op); ``commit_changes`` takes the per-file actors
for the repository it is about to commit with ``take_for_commit``. The
baseline only moves on a successful export, so a failed export keeps its
authors for the retry.
"""

import json
import logging
import os
import re
import threading
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

STATE_VECTORS_FILE = "document_state_vectors.json"
USERS_MAP_KEY = "users"
MAX_FILE_LINES = 50
TRAILER_DOMAIN = "relay.invalid"
# Baselines change on nearly every export; write them at most this often.
# A crash loses at most this window, which only re-credits those writers once.
SAVE_INTERVAL_S = 30.0


def _read_varuint(data: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7


def decode_state_vector(data: bytes) -> Dict[int, int]:
    """Decode a v1-encoded Yjs state vector into {clientID: clock}."""
    result: Dict[int, int] = {}
    if not data:
        return result
    count, pos = _read_varuint(data, 0)
    for _ in range(count):
        client, pos = _read_varuint(data, pos)
        clock, pos = _read_varuint(data, pos)
        result[client] = clock
    return result


def _actor_rank(actor: str) -> int:
    # Same precedence as the relay (mcp/provenance.rs client_actor_map):
    # human beats ai beats a legacy raw user id.
    if actor.startswith("human:"):
        return 2
    if actor.startswith("ai:"):
        return 1
    return 0


def client_actor_map(doc) -> Dict[int, str]:
    """Reverse the doc's "users" map into {clientID: actor key}."""
    from pycrdt import Array, Map

    result: Dict[int, str] = {}
    if USERS_MAP_KEY not in doc.keys():
        return result
    users = doc.get(USERS_MAP_KEY, type=Map)
    for actor, entry in users.items():
        if not isinstance(entry, Map) or "ids" not in entry:
            continue
        ids = entry["ids"]
        if not isinstance(ids, Array):
            continue
        for raw in ids:
            try:
                client = int(raw)
            except (TypeError, ValueError):
                continue
            existing = result.get(client)
            if existing is None or _actor_rank(actor) > _actor_rank(existing):
                result[client] = actor
    return result


def display_actor(actor: str) -> str:
    """Readable label: the bare name for humans, the raw key for AI, and
    ``relay-user:<id>`` for an Obsidian/Relay.md user id."""
    if actor.startswith("human:"):
        return actor[len("human:") :].strip() or "unknown"
    if actor.startswith("ai:"):
        return actor
    return f"relay-user:{actor}"


def _clean(text: str) -> str:
    # Commit message lines must not be split or forged by a self-reported name.
    return re.sub(r"[\x00-\x1f\x7f<>]", " ", text).strip()


def _trailer(label: str) -> str:
    local = re.sub(r"[^a-z0-9.]+", "-", label.lower()).strip("-.") or "unknown"
    return f"Co-authored-by: {label} <{local}@{TRAILER_DOMAIN}>"


def format_authors_body(changes: Dict[str, Set[str]]) -> str:
    """Commit message body for {repo-relative path: {actor keys}}; "" if none."""
    changes = {path: actors for path, actors in changes.items() if actors}
    if not changes:
        return ""

    labels: Dict[str, str] = {}
    for actors in changes.values():
        for actor in actors:
            labels[actor] = _clean(display_actor(actor)) or "unknown"
    ordered = sorted(set(labels.values()), key=str.lower)

    lines: List[str] = [f"Authors: {', '.join(ordered)}", ""]
    paths = sorted(changes)
    for path in paths[:MAX_FILE_LINES]:
        names = sorted({labels[a] for a in changes[path]}, key=str.lower)
        lines.append(f"- {_clean(path)}: {', '.join(names)}")
    if len(paths) > MAX_FILE_LINES:
        lines.append(f"- ... and {len(paths) - MAX_FILE_LINES} more files")
    lines.append("")
    lines.extend(_trailer(label) for label in ordered)
    return "\n".join(lines)


class AuthorTracker:
    """Per-document state-vector baselines plus per-repo pending authors."""

    def __init__(self):
        self._lock = threading.Lock()
        # relay_id -> doc_id -> {clientID(str): clock}; persisted
        self._baselines: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._loaded_relays: Set[str] = set()
        self._dirty: Set[str] = set()
        self._last_save: Dict[str, float] = {}
        # doc_id -> (state vector, clientID -> actor) from the latest fetch
        self._candidates: Dict[Tuple[str, str], Tuple[Dict[int, int], Dict[int, str]]] = {}
        # repo_key -> repo-relative path -> actor keys, awaiting a commit
        self._pending: Dict[str, Dict[str, Set[str]]] = {}

    # --- persistence -----------------------------------------------------

    def load(self, relay_id: str, state_dir: str):
        """Load baselines once per relay. Later calls are no-ops so a reload
        of the other state files cannot roll baselines back."""
        with self._lock:
            if relay_id in self._loaded_relays:
                return
            self._loaded_relays.add(relay_id)
            path = os.path.join(state_dir, STATE_VECTORS_FILE)
            baselines: Dict[str, Dict[str, int]] = {}
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        baselines = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading state vectors for relay {relay_id}: {e}")
            self._baselines.setdefault(relay_id, {}).update(baselines)

    def save(self, relay_id: str, state_dir: str, force: bool = False):
        with self._lock:
            if relay_id not in self._loaded_relays:
                return  # never loaded: writing now could clobber the file
            if relay_id not in self._dirty:
                return
            now = time.monotonic()
            if (
                not force
                and now - self._last_save.get(relay_id, -SAVE_INTERVAL_S) < SAVE_INTERVAL_S
            ):
                return
            self._dirty.discard(relay_id)
            self._last_save[relay_id] = now
            data = json.dumps(self._baselines.get(relay_id, {}))
        try:
            os.makedirs(state_dir, exist_ok=True)
            path = os.path.join(state_dir, STATE_VECTORS_FILE)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.write(data)
            os.replace(tmp, path)
        except Exception as e:
            with self._lock:
                self._dirty.add(relay_id)
            logger.error(f"Error saving state vectors for relay {relay_id}: {e}")

    # --- recording -------------------------------------------------------

    def observe(self, relay_id: str, doc_id: str, doc):
        """Remember a freshly fetched doc's state vector and actor map."""
        state_vector = decode_state_vector(doc.get_state())
        actors = client_actor_map(doc)
        with self._lock:
            self._candidates[(relay_id, doc_id)] = (state_vector, actors)

    def _changed_actors(
        self,
        relay_id: str,
        doc_id: str,
        state_vector: Dict[int, int],
        actors: Dict[int, str],
        is_new_file: bool,
    ) -> Set[str]:
        baseline = self._baselines.get(relay_id, {}).get(doc_id)
        if baseline is None and not is_new_file:
            # First sight of an existing file (e.g. right after deploy): no
            # reference point, so claim nothing rather than everyone ever.
            return set()
        baseline = baseline or {}
        changed: Set[str] = set()
        for client, clock in state_vector.items():
            if clock > baseline.get(str(client), 0) and client in actors:
                changed.add(actors[client])
        return changed

    def _set_baseline(self, relay_id, doc_id, state_vector, actors):
        # Only registered clients are kept: they are all attribution needs,
        # and the server's own clientID or a client that registers later
        # (its writes are then credited at its first registered export)
        # would only grow the file.
        self._baselines.setdefault(relay_id, {})[doc_id] = {
            str(client): clock for client, clock in state_vector.items() if client in actors
        }
        self._dirty.add(relay_id)

    def confirm(
        self,
        repo_key: str,
        relay_id: str,
        doc_id: str,
        path: str,
        is_new_file: bool = False,
    ) -> Set[str]:
        """The doc's export reached disk at repo-relative ``path``: attribute
        the clients that wrote since the baseline, then move the baseline."""
        with self._lock:
            candidate = self._candidates.pop((relay_id, doc_id), None)
            if candidate is None:
                return set()
            state_vector, actors = candidate
            changed = self._changed_actors(relay_id, doc_id, state_vector, actors, is_new_file)
            if changed:
                self._pending.setdefault(repo_key, {}).setdefault(path, set()).update(changed)
            self._set_baseline(relay_id, doc_id, state_vector, actors)
            return changed

    def advance(self, relay_id: str, doc_id: str):
        """The fetched content equals what is already exported: nobody's
        writes are visible in git, so just move the baseline."""
        with self._lock:
            candidate = self._candidates.pop((relay_id, doc_id), None)
            if candidate is None:
                return
            state_vector, actors = candidate
            self._set_baseline(relay_id, doc_id, state_vector, actors)

    def forget(self, relay_id: str, doc_id: str):
        """Drop an unused candidate (e.g. the export failed) so it cannot be
        confirmed later against newer content."""
        with self._lock:
            self._candidates.pop((relay_id, doc_id), None)

    # --- committing ------------------------------------------------------

    def take_for_commit(
        self, repo_key: str, staged_paths: Iterable[str], dirty_paths: Iterable[str] = ()
    ) -> Dict[str, Set[str]]:
        """Remove and return the authors of ``staged_paths``. Entries for
        paths written after staging (``dirty_paths``) stay for the next
        commit; anything else is stale and dropped."""
        staged = set(staged_paths)
        dirty = set(dirty_paths)
        with self._lock:
            pending = self._pending.pop(repo_key, {})
            taken = {p: a for p, a in pending.items() if p in staged}
            keep = {p: a for p, a in pending.items() if p not in staged and p in dirty}
            if keep:
                self._pending[repo_key] = keep
            return taken

    def rename(self, repo_key: str, old_path: str, new_path: str):
        """A file moved before its authors were committed: carry them over."""
        with self._lock:
            repo = self._pending.get(repo_key)
            if repo and old_path in repo:
                repo.setdefault(new_path, set()).update(repo.pop(old_path))

    def restore(self, repo_key: str, changes: Dict[str, Set[str]]):
        """Put authors back after a failed commit."""
        if not changes:
            return
        with self._lock:
            repo = self._pending.setdefault(repo_key, {})
            for path, actors in changes.items():
                repo.setdefault(path, set()).update(actors)
