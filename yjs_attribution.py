#!/usr/bin/env python3
"""Which Yjs client inserted each character of a doc's "contents" text.

Every Yjs item carries the ID (clientID, clock) of the client that created it,
but pycrdt does not expose item IDs. This module decodes the doc's v1 update
(the same bytes the relay serves) and re-runs YATA integration for the items of
the root "contents" Y.Text only, which yields the visible text together with
the ID of every character.

The result is only trusted when the reconstructed text equals the text pycrdt
itself produces; on any mismatch or decoding problem callers get ``None`` and
fall back to per-file attribution. Integration follows Yjs' Item.integrate
(the same algorithm yrs implements), and YATA's result does not depend on the
order in which causally-ready items are integrated.
"""

import bisect
import logging
import struct
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ID = Tuple[int, int]  # (client, clock)

_BIT8 = 0x80  # has origin (left)
_BIT7 = 0x40  # has right origin
_BIT6 = 0x20  # has parentSub
_BITS5 = 0x1F  # content ref

_CONTENT_DELETED = 1
_CONTENT_STRING = 4
_YXML_ELEMENT = 3
_YXML_HOOK = 5


class _Decoder:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def u8(self) -> int:
        b = self.data[self.pos]
        self.pos += 1
        return b

    def varuint(self) -> int:
        result = 0
        shift = 0
        while True:
            b = self.u8()
            result |= (b & 0x7F) << shift
            if b < 0x80:
                return result
            shift += 7

    def varint(self) -> int:
        b = self.u8()
        negative = b & 0x40
        result = b & 0x3F
        shift = 6
        while b & 0x80:
            b = self.u8()
            result |= (b & 0x7F) << shift
            shift += 7
        return -result if negative else result

    def raw(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError("truncated update")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def varbytes(self) -> bytes:
        return self.raw(self.varuint())

    def varstring(self) -> str:
        return self.varbytes().decode("utf-8")

    def any(self):
        t = self.u8()
        if t == 127:
            return None  # undefined
        if t == 126:
            return None
        if t == 125:
            return self.varint()
        if t == 124:
            return struct.unpack(">f", self.raw(4))[0]
        if t == 123:
            return struct.unpack(">d", self.raw(8))[0]
        if t == 122:
            return struct.unpack(">q", self.raw(8))[0]
        if t == 121:
            return False
        if t == 120:
            return True
        if t == 119:
            return self.varstring()
        if t == 118:
            return {self.varstring(): self.any() for _ in range(self.varuint())}
        if t == 117:
            return [self.any() for _ in range(self.varuint())]
        if t == 116:
            return self.varbytes()
        raise ValueError(f"unknown Any type {t}")


class _Item:
    __slots__ = (
        "client",
        "clock",
        "length",
        "origin",
        "right_origin",
        "parent",
        "parent_sub",
        "kind",
        "units",
        "left",
        "right",
        "integrated",
    )

    def __init__(
        self, client, clock, length, origin, right_origin, parent, parent_sub, kind, units
    ):
        self.client = client
        self.clock = clock
        self.length = length
        self.origin = origin
        self.right_origin = right_origin
        self.parent = parent  # root name (str), parent type ID (tuple) or None
        self.parent_sub = parent_sub
        self.kind = kind
        self.units = units  # UTF-16 code units for string content, else None
        self.left = None
        self.right = None
        self.integrated = False


class _Store:
    """All structs of the update, per client, sorted by clock."""

    def __init__(self):
        self.structs: Dict[int, List[object]] = {}
        self.starts: Dict[int, List[int]] = {}

    def add(self, client: int, s):
        self.structs.setdefault(client, []).append(s)
        self.starts.setdefault(client, []).append(s.clock)

    def find(self, ident: ID):
        client, clock = ident
        starts = self.starts.get(client)
        if not starts:
            return None
        i = bisect.bisect_right(starts, clock) - 1
        if i < 0:
            return None
        s = self.structs[client][i]
        if clock >= s.clock + s.length:
            return None
        return s

    def split(self, item: _Item, diff: int) -> _Item:
        """Split ``item`` so that the right part starts ``diff`` units in."""
        right = _Item(
            item.client,
            item.clock + diff,
            item.length - diff,
            (item.client, item.clock + diff - 1),
            item.right_origin,
            item.parent,
            item.parent_sub,
            item.kind,
            item.units[diff:] if item.units is not None else None,
        )
        if item.units is not None:
            item.units = item.units[:diff]
        item.length = diff
        right.integrated = item.integrated
        if item.integrated:
            right.left = item
            right.right = item.right
            if item.right is not None:
                item.right.left = right
            item.right = right
        lst = self.structs[item.client]
        i = bisect.bisect_left(self.starts[item.client], item.clock)
        lst.insert(i + 1, right)
        self.starts[item.client].insert(i + 1, right.clock)
        return right

    def clean_end(self, ident: ID) -> Optional[_Item]:
        s = self.find(ident)
        if not isinstance(s, _Item):
            return None
        if ident[1] != s.clock + s.length - 1:
            self.split(s, ident[1] - s.clock + 1)
        return s

    def clean_start(self, ident: ID) -> Optional[_Item]:
        s = self.find(ident)
        if not isinstance(s, _Item):
            return None
        if ident[1] != s.clock:
            return self.split(s, ident[1] - s.clock)
        return s


class _Gap:
    __slots__ = ("clock", "length")

    def __init__(self, clock, length):
        self.clock = clock
        self.length = length


def _utf16_units(s: str) -> List[int]:
    b = s.encode("utf-16-le", "surrogatepass")
    return list(struct.unpack(f"<{len(b) // 2}H", b))


def _read_content(dec: _Decoder, ref: int):
    """Return (length, units-or-None) and consume the content."""
    if ref == _CONTENT_DELETED:
        return dec.varuint(), None
    if ref == 2:  # JSON
        n = dec.varuint()
        for _ in range(n):
            dec.varstring()
        return n, None
    if ref == 3:  # binary
        dec.varbytes()
        return 1, None
    if ref == _CONTENT_STRING:
        units = _utf16_units(dec.varstring())
        return len(units), units
    if ref == 5:  # embed
        dec.varstring()
        return 1, None
    if ref == 6:  # format
        dec.varstring()
        dec.varstring()
        return 1, None
    if ref == 7:  # type
        type_ref = dec.varuint()
        if type_ref in (_YXML_ELEMENT, _YXML_HOOK):
            dec.varstring()
        return 1, None
    if ref == 8:  # any
        n = dec.varuint()
        for _ in range(n):
            dec.any()
        return n, None
    if ref == 9:  # subdoc
        dec.varstring()
        dec.any()
        return 1, None
    raise ValueError(f"unknown content ref {ref}")


def _decode(update: bytes):
    dec = _Decoder(update)
    store = _Store()
    for _ in range(dec.varuint()):
        n_structs = dec.varuint()
        client = dec.varuint()
        clock = dec.varuint()
        for _ in range(n_structs):
            info = dec.u8()
            ref = info & _BITS5
            if ref == 0:  # GC
                length = dec.varuint()
                store.add(client, _Gap(clock, length))
            elif ref == 10:  # skip
                length = dec.varuint()
            else:
                origin = (dec.varuint(), dec.varuint()) if info & _BIT8 else None
                right_origin = (dec.varuint(), dec.varuint()) if info & _BIT7 else None
                parent = None
                parent_sub = None
                if not info & (_BIT8 | _BIT7):
                    if dec.varuint() == 1:
                        parent = dec.varstring()
                    else:
                        parent = (dec.varuint(), dec.varuint())
                    if info & _BIT6:
                        parent_sub = dec.varstring()
                length, units = _read_content(dec, ref)
                store.add(
                    client,
                    _Item(
                        client, clock, length, origin, right_origin, parent, parent_sub, ref, units
                    ),
                )
            clock += length
    deletes: Dict[int, List[Tuple[int, int]]] = {}
    if dec.pos < len(update):
        for _ in range(dec.varuint()):
            client = dec.varuint()
            ranges = []
            for _ in range(dec.varuint()):
                start = dec.varuint()
                ranges.append((start, start + dec.varuint()))
            deletes.setdefault(client, []).extend(ranges)
    for ranges in deletes.values():
        ranges.sort()
    return store, deletes


def _resolve_parents(store: _Store):
    """Copy parent info from origins like Yjs does; items whose origin is
    missing or garbage-collected get no parent (Yjs turns them into GC)."""
    for structs in store.structs.values():
        for s in structs:
            if not isinstance(s, _Item) or s.parent is not None:
                continue
            chain = [s]
            cur = s
            resolved = None
            while True:
                ref = cur.origin or cur.right_origin
                nxt = store.find(ref) if ref else None
                if not isinstance(nxt, _Item):
                    break
                if nxt.parent is not None:
                    resolved = (nxt.parent, nxt.parent_sub)
                    break
                if nxt in chain:
                    break
                chain.append(nxt)
                cur = nxt
            if resolved is None:
                resolved = ("", None)  # orphan: never part of any type
            for c in chain:
                c.parent, c.parent_sub = resolved


class _Text:
    def __init__(self):
        self.start: Optional[_Item] = None


def _integrate(store: _Store, text: _Text, item: _Item):
    left = store.clean_end(item.origin) if item.origin else None
    right = store.clean_start(item.right_origin) if item.right_origin else None
    if (left is None and (right is None or right.left is not None)) or (
        left is not None and left.right is not right
    ):
        o = left.right if left is not None else text.start
        conflicting = set()
        before_origin = set()
        while o is not None and o is not right:
            before_origin.add(id(o))
            conflicting.add(id(o))
            if item.origin == o.origin:
                if o.client < item.client:
                    left = o
                    conflicting.clear()
                elif item.right_origin == o.right_origin:
                    break
            elif o.origin is not None:
                oi = store.find(o.origin)
                if oi is not None and id(oi) in before_origin:
                    if id(oi) not in conflicting:
                        left = o
                        conflicting.clear()
                else:
                    break
            else:
                break
            o = o.right
    item.left = left
    if left is not None:
        item.right = left.right
        left.right = item
    else:
        item.right = text.start
        text.start = item
    if item.right is not None:
        item.right.left = item
    item.integrated = True


def _ready_deps(store: _Store, item: _Item):
    deps = []
    for ref in (item.origin, item.right_origin):
        if ref:
            dep = store.find(ref)
            if isinstance(dep, _Item) and not dep.integrated:
                deps.append(dep)
    return deps


def _integrate_with_deps(store: _Store, text: _Text, item: _Item, root: str):
    stack = [item]
    while stack:
        top = stack[-1]
        if top.integrated:
            stack.pop()
            continue
        deps = [d for d in _ready_deps(store, top) if d.parent == root]
        if deps:
            # One at a time, so the stack is exactly the dependency path.
            if deps[0] in stack:
                raise ValueError("cyclic item dependencies")
            stack.append(deps[0])
            continue
        stack.pop()
        _integrate(store, text, top)


def character_ids(update: bytes, root: str = "contents") -> Optional[Tuple[str, List[ID]]]:
    """Visible text of the root Y.Text ``root`` and, per character, the ID
    of the item that inserted it. ``None`` if the update cannot be read."""
    try:
        store, deletes = _decode(update)
        _resolve_parents(store)
        text = _Text()
        while True:
            # Re-collect each round: splits create new, unintegrated parts.
            pending = [
                s
                for client in sorted(store.structs)
                for s in store.structs[client]
                if isinstance(s, _Item)
                and not s.integrated
                and s.parent == root
                and s.parent_sub is None
            ]
            if not pending:
                break
            for item in pending:
                if not item.integrated:
                    _integrate_with_deps(store, text, item, root)

        units: List[int] = []
        owners: List[ID] = []
        o = text.start
        while o is not None:
            if o.kind == _CONTENT_STRING and o.units:
                ranges = deletes.get(o.client, [])
                for offset, unit in enumerate(o.units):
                    clock = o.clock + offset
                    i = bisect.bisect_right(ranges, (clock, float("inf"))) - 1
                    if i >= 0 and ranges[i][0] <= clock < ranges[i][1]:
                        continue
                    units.append(unit)
                    owners.append((o.client, clock))
            o = o.right

        chars: List[str] = []
        char_ids: List[ID] = []
        i = 0
        while i < len(units):
            u = units[i]
            if 0xD800 <= u < 0xDC00 and i + 1 < len(units) and 0xDC00 <= units[i + 1] < 0xE000:
                chars.append(chr(0x10000 + ((u - 0xD800) << 10) + (units[i + 1] - 0xDC00)))
                char_ids.append(owners[i])
                i += 2
            else:
                chars.append(chr(u))
                char_ids.append(owners[i])
                i += 1
        return "".join(chars), char_ids
    except Exception as e:  # malformed or unsupported: caller falls back
        logger.warning(f"Could not reconstruct character IDs: {e}")
        return None


def line_owners(
    text: str,
    char_ids: List[ID],
    baseline: Dict[str, int],
    actors: Dict[int, str],
) -> List[Optional[str]]:
    """Per line of ``text`` (split with keepends), the actor who inserted most
    of the line's characters written after ``baseline``; None if nobody."""
    owners: List[Optional[str]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        counts: Dict[str, int] = {}
        for client, clock in char_ids[pos : pos + len(line)]:
            actor = actors.get(client)
            if actor is not None and clock >= baseline.get(str(client), 0):
                counts[actor] = counts.get(actor, 0) + 1
        pos += len(line)
        owners.append(min(counts, key=lambda a: (-counts[a], a)) if counts else None)
    return owners
