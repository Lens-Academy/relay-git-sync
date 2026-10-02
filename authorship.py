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

Per character, yjs_attribution.py recovers the clientID that inserted it, so
each line of an exported file can be given to the actor who wrote most of its
new characters; the commit timer turns that into one commit per author (see
PersistenceManager._write_author_commits) so ``git blame`` shows who wrote
each line.

Flow: RelayClient calls ``observe`` with each fetched Y.Doc; the sync engine
calls ``confirm`` once that doc's export reached disk (or ``advance`` when the
export turned out to be a no-op); ``commit_changes`` takes the pending entries
of the files it is about to commit with ``take_for_commit`` and then moves
each doc's baseline with ``committed``. Baselines are the state at the last
commit, so everything written since is attributed even across several exports
or a restart, and a failed export or commit loses nothing.
"""

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from yjs_attribution import character_ids, line_owners

logger = logging.getLogger(__name__)

STATE_VECTORS_FILE = "document_state_vectors.json"
USERS_MAP_KEY = "users"
MAX_FILE_LINES = 50
# Bounds the message even if a doc's "users" map is flooded with names.
MAX_AUTHORS = 20
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


def actor_label(actor: str) -> str:
    """Display name safe for commit messages and git author fields. Git
    trims these characters from ident ends and rejects an empty name."""
    label = _clean(display_actor(actor)).strip(" .,:;\"'\\")
    return label if any(ch.isalnum() for ch in label) else "unknown"


def actor_email(label: str) -> str:
    local = re.sub(r"[^a-z0-9.]+", "-", label.lower()).strip("-.") or "unknown"
    return f"{local}@{TRAILER_DOMAIN}"


def _trailer(label: str) -> str:
    return f"Co-authored-by: {label} <{actor_email(label)}>"


def format_authors_body(changes: Dict[str, Set[str]]) -> str:
    """Commit message body for {repo-relative path: {actor keys}}; "" if none."""
    changes = {path: actors for path, actors in changes.items() if actors}
    if not changes:
        return ""

    labels: Dict[str, str] = {}
    for actors in changes.values():
        for actor in actors:
            labels[actor] = actor_label(actor)
    ordered = sorted(set(labels.values()), key=str.lower)

    shown = ordered[:MAX_AUTHORS]
    more = f" and {len(ordered) - MAX_AUTHORS} more" if len(ordered) > MAX_AUTHORS else ""
    lines: List[str] = [f"Authors: {', '.join(shown)}{more}", ""]
    paths = sorted(changes)
    for path in paths[:MAX_FILE_LINES]:
        names = sorted({labels[a] for a in changes[path]}, key=str.lower)[:MAX_AUTHORS]
        lines.append(f"- {_clean(path)}: {', '.join(names)}")
    if len(paths) > MAX_FILE_LINES:
        lines.append(f"- ... and {len(paths) - MAX_FILE_LINES} more files")
    lines.append("")
    lines.extend(_trailer(label) for label in shown)
    return "\n".join(lines)


@dataclass
class PendingFile:
    """An exported file whose authors still await a commit."""

    doc_id: str
    relay_id: str
    state_vector: Dict[str, int]  # baseline to store once committed
    actors: Set[str]  # everyone who wrote since the last commit
    content: str  # the exported text the line owners refer to
    owners: Optional[List[Optional[str]]]  # per line; None = unknown
    is_new: bool = False  # the export created the file


class AuthorTracker:
    """Per-document committed baselines plus per-repo pending files."""

    def __init__(self):
        self._lock = threading.Lock()
        # relay_id -> doc_id -> {clientID(str): clock} at the last commit; persisted
        self._baselines: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._loaded_relays: Set[str] = set()
        self._dirty: Set[str] = set()
        self._last_save: Dict[str, float] = {}
        # (relay_id, doc_id) -> (doc update, state vector, clientID -> actor)
        self._candidates: Dict[Tuple[str, str], Tuple[bytes, Dict[int, int], Dict[int, str]]] = {}
        # repo_key -> repo-relative path -> pending file
        self._pending: Dict[str, Dict[str, PendingFile]] = {}

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
            if relay_id not in self._loaded_relays or relay_id not in self._dirty:
                return  # never loaded (would clobber the file) or nothing new
            now = time.monotonic()
            last = self._last_save.get(relay_id)
            if not force and last is not None and now - last < SAVE_INTERVAL_S:
                return
            self._dirty.discard(relay_id)
            data = json.dumps(self._baselines.get(relay_id, {}))
        try:
            os.makedirs(state_dir, exist_ok=True)
            path = os.path.join(state_dir, STATE_VECTORS_FILE)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.write(data)
            os.replace(tmp, path)
            with self._lock:
                self._last_save[relay_id] = now
        except Exception as e:
            with self._lock:
                self._dirty.add(relay_id)
            logger.error(f"Error saving state vectors for relay {relay_id}: {e}")

    # --- recording -------------------------------------------------------

    def observe(self, relay_id: str, doc_id: str, doc):
        """Remember a freshly fetched doc until its export is settled."""
        update = doc.get_update()
        state_vector = decode_state_vector(doc.get_state())
        actors = client_actor_map(doc)
        with self._lock:
            self._candidates[(relay_id, doc_id)] = (update, state_vector, actors)

    def _set_baseline(self, relay_id, doc_id, state_vector: Dict[str, int]):
        self._baselines.setdefault(relay_id, {})[doc_id] = state_vector
        self._dirty.add(relay_id)

    @staticmethod
    def _registered(state_vector: Dict[int, int], actors: Dict[int, str]) -> Dict[str, int]:
        # Only registered clients matter for attribution; keeping the rest
        # (the server's own clientID, unregistered sessions) only grows the file.
        return {str(c): clock for c, clock in state_vector.items() if c in actors}

    def confirm(
        self,
        repo_key: str,
        relay_id: str,
        doc_id: str,
        path: str,
        content: Optional[str] = None,
        is_new_file: bool = False,
    ) -> Set[str]:
        """The doc's export reached disk at repo-relative ``path``: record who
        wrote since the last commit, and per line of ``content``, whom."""
        with self._lock:
            candidate = self._candidates.pop((relay_id, doc_id), None)
            if candidate is None:
                return set()
            baseline = self._baselines.get(relay_id, {}).get(doc_id)
        update, state_vector, actors = candidate
        if baseline is None and not is_new_file:
            # First sight of an existing file (e.g. right after deploy): no
            # reference point, so claim nothing rather than everyone ever.
            changed: Set[str] = set()
            owners = None
        else:
            baseline = baseline or {}
            changed = {
                actors[c]
                for c, clock in state_vector.items()
                if c in actors and clock > baseline.get(str(c), 0)
            }
            owners = None
            if changed and content is not None:
                chars = character_ids(update)
                if chars is not None and chars[0] == content:
                    owners = line_owners(content, chars[1], baseline, actors)
                elif chars is not None:
                    logger.warning(
                        f"Text reconstruction differs for {doc_id}; per-file authors only"
                    )
        entry = PendingFile(
            doc_id,
            relay_id,
            self._registered(state_vector, actors),
            changed,
            content or "",
            owners,
            is_new_file,
        )
        with self._lock:
            # Newer exports supersede older ones: they are computed against
            # the same committed baseline, so they already include them.
            self._pending.setdefault(repo_key, {})[path] = entry
        return changed

    def advance(self, relay_id: str, doc_id: str):
        """The fetched content equals what is already exported. With nothing
        pending for the doc, git already has it: move the baseline."""
        with self._lock:
            candidate = self._candidates.pop((relay_id, doc_id), None)
            if candidate is None:
                return
            for repo in self._pending.values():
                if any(e.doc_id == doc_id and e.relay_id == relay_id for e in repo.values()):
                    return
            _, state_vector, actors = candidate
            self._set_baseline(relay_id, doc_id, self._registered(state_vector, actors))

    def forget(self, relay_id: str, doc_id: str):
        """Drop an unused candidate (e.g. the export failed)."""
        with self._lock:
            self._candidates.pop((relay_id, doc_id), None)

    # --- committing ------------------------------------------------------

    def take_for_commit(
        self, repo_key: str, staged_paths: Iterable[str], dirty_paths: Iterable[str] = ()
    ) -> Dict[str, PendingFile]:
        """Remove and return the pending files among ``staged_paths``. Files
        written again after staging (``dirty_paths``) stay pending: their
        newest export is not in this commit. Anything else is stale."""
        staged = set(staged_paths)
        dirty = set(dirty_paths)
        with self._lock:
            pending = self._pending.pop(repo_key, {})
            taken = {p: e for p, e in pending.items() if p in staged and p not in dirty}
            keep = {p: e for p, e in pending.items() if p in dirty}
            if keep:
                self._pending[repo_key] = keep
            return taken

    def committed(self, entries: Iterable[PendingFile]):
        """These exports are in git now: they become the docs' baselines."""
        with self._lock:
            for e in entries:
                self._set_baseline(e.relay_id, e.doc_id, e.state_vector)

    def rename(self, repo_key: str, old_path: str, new_path: str):
        """A file moved before its authors were committed: carry them over."""
        with self._lock:
            repo = self._pending.get(repo_key)
            if repo and old_path in repo:
                repo[new_path] = repo.pop(old_path)

    def restore(self, repo_key: str, entries: Dict[str, PendingFile]):
        """Put entries back after a failed commit (newer ones win)."""
        if not entries:
            return
        with self._lock:
            repo = self._pending.setdefault(repo_key, {})
            for path, entry in entries.items():
                repo.setdefault(path, entry)
