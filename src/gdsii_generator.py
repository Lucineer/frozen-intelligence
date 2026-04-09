#!/usr/bin/env python3
"""GDSII stream format generator for mask-locked inference chips.

Generates simplified GDSII binary for physical layout visualization.
Real GDSII is complex (200+ page spec) — this generates valid structure
for weight bank tiles, routing channels, and pad frames.
"""
import struct, math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import IntEnum


class GDSRecordType(IntEnum):
    HEADER = 0x0002
    BGNLIB = 0x0102
    LIBNAME = 0x0206
    UNITS = 0x0305
    ENDLIB = 0x0400
    BGNSTR = 0x0502
    STRNAME = 0x0606
    ENDSTR = 0x0700
    BOUNDARY = 0x0800
    PATH = 0x0900
    SREF = 0x0A00
    AREF = 0x0B00
    TEXT = 0x0C00
    LAYER = 0x0D02
    DATATYPE = 0x0E02
    WIDTH = 0x0F03
    XY = 0x1003
    ENDEL = 0x1100
    SNAME = 0x1206
    COLROW = 0x1302
    STRANS = 0x1A01
    MAG = 0x1B05
    ANGLE = 0x1C05


class GDSType(IntEnum):
    NODATA = 0
    BITARRAY = 1
    TWO_BYTE_INT = 2
    FOUR_BYTE_INT = 3
    EIGHT_BYTE_REAL = 4
    FLOAT = 5


class Layer:
    GATE = 1       # Poly gate
    METAL1 = 2     # First metal
    METAL2 = 3     # Second metal
    VIA1 = 4       # Metal1-Metal2 via
    CONTACT = 5    # Contact to silicon
    ACTIVE = 6     # Active region
    N_WELL = 7     # N-well
    P_WELL = 8     # P-well
    PAD = 9        # Bonding pad
    TEXT = 10      # Annotation text


class GDSDatabase:
    """GDSII stream database builder."""

    def __init__(self, lib_name: str = "frozen_intelligence",
                 db_unit_m: float = 1e-9, user_unit_m: float = 1e-6):
        self.lib_name = lib_name
        self.db_per_user = int(user_unit_m / db_unit_m)
        self.m_per_db = db_unit_m
        self.structures: List['GDSStructure'] = []

    def add_structure(self, name: str) -> 'GDSStructure':
        s = GDSStructure(name)
        self.structures.append(s)
        return s

    def to_bytes(self) -> bytes:
        buf = bytearray()
        # Library header
        buf += self._record(GDSRecordType.HEADER, 600)
        buf += self._record(GDSRecordType.BGNLIB, (2024, 1, 1, 0, 0, 0, 2024, 1, 1, 0, 0, 0))
        buf += self._record(GDSRecordType.LIBNAME, self.lib_name)
        buf += self._record(GDSRecordType.UNITS, (self.db_per_user, 1e-6 / self.m_per_db))
        for s in self.structures:
            buf += s.to_bytes()
        buf += self._record(GDSRecordType.ENDLIB, None)
        return bytes(buf)

    def _record(self, rtype: int, data) -> bytes:
        buf = bytearray()
        if data is None:
            buf += struct.pack(">HH", 4, rtype)
        elif isinstance(data, int):
            buf += struct.pack(">HHh", 6, rtype, data)
        elif isinstance(data, tuple) and all(isinstance(x, int) for x in data):
            if all(-128 <= x <= 127 for x in data):
                buf += struct.pack(">HH", 4, rtype)
                for x in data:
                    buf += struct.pack(">h", x)
                buf[0:2] = struct.pack(">H", len(buf))
            elif all(-32768 <= x <= 32767 for x in data):
                buf += struct.pack(">HH", 4, rtype)
                for x in data:
                    buf += struct.pack(">h", x)
                buf[0:2] = struct.pack(">H", len(buf))
            else:
                buf += struct.pack(">HH", 4, rtype)
                for x in data:
                    buf += struct.pack(">i", x)
                buf[0:2] = struct.pack(">H", len(buf))
        elif isinstance(data, str):
            encoded = data.encode("ascii")
            if len(encoded) % 2 != 0:
                encoded += b"\\x00"
            buf += struct.pack(">HH", 4 + len(encoded), rtype)
            buf += encoded
        elif isinstance(data, float):
            buf += struct.pack(">HH", 12, rtype)
            buf += struct.pack(">d", data)
        return bytes(buf)


class GDSStructure:
    """GDSII structure (cell)."""

    def __init__(self, name: str):
        self.name = name
        self.elements: List[bytes] = []
        self.references: List[Tuple[str, int, int, float]] = []

    def add_boundary(self, layer: int, datatype: int = 0,
                     coords: List[Tuple[int, int]] = None) -> 'GDSStructure':
        """Add rectangular boundary."""
        if coords is None:
            coords = [(0, 0), (1000, 0), (1000, 1000), (0, 1000), (0, 0)]
        buf = bytearray()
        buf += struct.pack(">HH", 4, GDSRecordType.BOUNDARY)
        buf += struct.pack(">HHh", 6, GDSRecordType.LAYER, layer)
        buf += struct.pack(">HHh", 6, GDSRecordType.DATATYPE, datatype)
        # XY record
        xy_data = b""
        for x, y in coords:
            xy_data += struct.pack(">ii", x, y)
        buf += struct.pack(">HH", 4 + len(xy_data), GDSRecordType.XY)
        buf += xy_data
        buf += struct.pack(">HH", 4, GDSRecordType.ENDEL)
        self.elements.append(bytes(buf))
        return self

    def add_rect(self, layer: int, x: int, y: int, w: int, h: int) -> 'GDSStructure':
        """Add rectangle (closed boundary)."""
        coords = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
        return self.add_boundary(layer, 0, coords)

    def add_path(self, layer: int, width: int,
                 coords: List[Tuple[int, int]]) -> 'GDSStructure':
        """Add wire path."""
        buf = bytearray()
        buf += struct.pack(">HH", 4, GDSRecordType.PATH)
        buf += struct.pack(">HHh", 6, GDSRecordType.LAYER, layer)
        buf += struct.pack(">HHh", 6, GDSRecordType.DATATYPE, 0)
        buf += struct.pack(">HHi", 8, GDSRecordType.WIDTH, width)
        xy_data = b""
        for x, y in coords:
            xy_data += struct.pack(">ii", x, y)
        buf += struct.pack(">HH", 4 + len(xy_data), GDSRecordType.XY)
        buf += xy_data
        buf += struct.pack(">HH", 4, GDSRecordType.ENDEL)
        self.elements.append(bytes(buf))
        return self

    def add_sref(self, sname: str, x: int, y: int, mag: float = 1.0) -> 'GDSStructure':
        """Add structure reference."""
        self.references.append((sname, x, y, mag))
        return self

    def add_text(self, layer: int, x: int, y: int, text: str) -> 'GDSStructure':
        """Add text annotation."""
        buf = bytearray()
        buf += struct.pack(">HH", 4, GDSRecordType.TEXT)
        buf += struct.pack(">HHh", 6, GDSRecordType.LAYER, layer)
        buf += struct.pack(">HHh", 6, GDSRecordType.DATATYPE, 0)
        encoded = text.encode("ascii")
        if len(encoded) % 2 != 0:
            encoded += b"\\x00"
        buf += struct.pack(">HH", 4 + len(encoded), GDSRecordType.SNAME)
        buf += encoded
        xy_data = struct.pack(">ii", x, y)
        buf += struct.pack(">HH", 4 + len(xy_data), GDSRecordType.XY)
        buf += xy_data
        buf += struct.pack(">HH", 4, GDSRecordType.ENDEL)
        self.elements.append(bytes(buf))
        return self

    def to_bytes(self) -> bytes:
        buf = bytearray()
        buf += struct.pack(">HH", 4, GDSRecordType.BGNSTR)
        buf += struct.pack(">HH", 4 + 32, GDSRecordType.BGNSTR)
        buf += struct.pack(">hhhhhhhhhhhhhhhh", 2024, 1, 1, 0, 0, 0, 2024, 1, 1, 0, 0, 0, 0, 0, 0, 0)
        name_enc = self.name.encode("ascii")
        if len(name_enc) % 2 != 0:
            name_enc += b"\\x00"
        buf += struct.pack(">HH", 4 + len(name_enc), GDSRecordType.STRNAME)
        buf += name_enc
        for elem in self.elements:
            buf += elem
        buf += struct.pack(">HH", 4, GDSRecordType.ENDSTR)
        return bytes(buf)


class ChipLayoutGenerator:
    """Generate GDSII layout for mask-locked chips."""

    def __init__(self, db: GDSDatabase):
        self.db = db
        self.tile_size_um = 100  # 100um tile pitch
        self.db_per_um = int(1e-6 / db.m_per_db)

    def _um(self, um: float) -> int:
        return int(um * self.db_per_um)

    def generate_weight_tile(self, name: str, rows: int = 4, cols: int = 4) -> GDSStructure:
        """Generate a weight bank tile layout."""
        s = self.db.add_structure(f"tile_{name}")
        tile = self.tile_size_um
        # Active area
        s.add_rect(Layer.ACTIVE, self._um(2), self._um(2),
                   self._um(tile - 4), self._um(tile - 4))
        # Gate poly grid (weight bits)
        pitch = self._um(tile - 4) / rows
        for i in range(rows):
            y = self._um(2) + int(i * pitch)
            s.add_rect(Layer.GATE, self._um(2), y,
                       self._um(tile - 4), max(1, int(pitch * 0.3)))
        # Metal1 routing
        for j in range(cols):
            x = self._um(2) + int(j * pitch)
            s.add_rect(Layer.METAL1, x, self._um(2),
                       max(1, int(pitch * 0.3)), self._um(tile - 4))
        # Metal2 power rails
        s.add_rect(Layer.METAL2, 0, 0, self._um(tile), self._um(1))
        s.add_rect(Layer.METAL2, 0, self._um(tile - 1), self._um(tile), self._um(1))
        # Via contacts at intersections
        for i in range(rows):
            for j in range(cols):
                x = self._um(2) + int(j * pitch)
                y = self._um(2) + int(i * pitch)
                via_size = max(1, int(pitch * 0.15))
                s.add_rect(Layer.VIA1, x, y, via_size, via_size)
        # Label
        s.add_text(Layer.TEXT, self._um(5), self._um(tile + 5), name)
        return s

    def generate_pad_ring(self, name: str, die_size_um: float,
                          num_pads: int = 40) -> GDSStructure:
        """Generate IO pad ring around die."""
        s = self.db.add_structure(f"padring_{name}")
        die = self._um(die_size_um)
        pad_size = self._um(50)
        pad_pitch = int((die - 2 * pad_size) / (num_pads // 4))
        # Top pads
        for i in range(num_pads // 4):
            x = pad_size + i * pad_pitch
            s.add_rect(Layer.PAD, x, 0, pad_size, pad_size)
        # Bottom pads
        for i in range(num_pads // 4):
            x = pad_size + i * pad_pitch
            s.add_rect(Layer.PAD, x, die - pad_size, pad_size, pad_size)
        # Left pads
        for i in range(num_pads // 4):
            y = pad_size + i * pad_pitch
            s.add_rect(Layer.PAD, 0, y, pad_size, pad_size)
        # Right pads
        for i in range(num_pads // 4):
            y = pad_size + i * pad_pitch
            s.add_rect(Layer.PAD, die - pad_size, y, pad_size, pad_size)
        # Seal ring
        s.add_rect(Layer.METAL2, 0, 0, die, self._um(5))
        s.add_rect(Layer.METAL2, 0, die - self._um(5), die, self._um(5))
        s.add_rect(Layer.METAL2, 0, 0, self._um(5), die)
        s.add_rect(Layer.METAL2, die - self._um(5), 0, self._um(5), die)
        return s

    def generate_top_level(self, name: str, tiles: List[str],
                           die_size_um: float = 2000) -> GDSStructure:
        """Generate top-level chip assembly."""
        s = self.db.add_structure(name)
        die = self._um(die_size_um)
        # Die boundary
        s.add_rect(Layer.ACTIVE, 0, 0, die, die)
        # Place tiles
        cols = int(math.sqrt(len(tiles)))
        if cols == 0:
            cols = 1
        rows = (len(tiles) + cols - 1) // cols
        tile_um = (die_size_um - 200) / max(cols, rows)  # Leave margin for pads
        for i, tile_name in enumerate(tiles):
            row = i // cols
            col = i % cols
            x = self._um(100 + col * tile_um)
            y = self._um(100 + row * tile_um)
            s.add_sref(f"tile_{tile_name}", x, y)
        # Pad ring
        s.add_sref(f"padring_{name}", 0, 0)
        # Power rails
        rail_w = self._um(2)
        for i in range(4):
            y = int(die * (i + 1) / 5)
            s.add_rect(Layer.METAL1, 0, y, die, rail_w)
        return s

    def save(self, filename: str):
        """Save GDSII to file."""
        data = self.db.to_bytes()
        with open(filename, "wb") as f:
            f.write(data)
        return len(data)

    def stats(self) -> Dict:
        """Layout statistics."""
        return {
            "structures": len(self.db.structures),
            "lib_name": self.db.lib_name,
            "db_per_user": self.db.db_per_user,
            "tile_size_um": self.tile_size_um,
            "resolution_nm": self.db.m_per_db * 1e9,
        }


def demo():
    print("=== GDSII Layout Generator ===\n")

    db = GDSDatabase("frozen_intelligence")
    gen = ChipLayoutGenerator(db)

    # Generate weight tiles
    tiles = ["router", "safety", "primary_cls", "feat_ext_a",
             "feat_ext_b", "audio_cmd", "sensor_fuse"]
    for name in tiles:
        gen.generate_weight_tile(name)
        print(f"  Generated tile: {name}")

    # Generate pad ring
    gen.generate_pad_ring("deckboss", 2000, 40)
    print(f"  Generated pad ring: deckboss (2000um, 40 pads)")

    # Generate top level
    gen.generate_top_level("deckboss_v1", tiles, 2000)
    print(f"  Generated top level: deckboss_v1")

    # Stats
    stats = gen.stats()
    print(f"\n  Structures: {stats['structures']}")
    print(f"  Resolution: {stats['resolution_nm']:.1f}nm")
    print(f"  Tile size: {stats['tile_size_um']}um")

    # Save
    size = gen.save("/tmp/frozen_intelligence.gds")
    print(f"  GDSII file: {size:,} bytes -> /tmp/frozen_intelligence.gds")

    # Verify file header
    with open("/tmp/frozen_intelligence.gds", "rb") as f:
        header = f.read(4)
        rec_size, rec_type = struct.unpack(">HH", header)
        print(f"\n  Header record: size={rec_size}, type=0x{rec_type:04X}")
        print(f"  Valid GDSII: {rec_type == GDSRecordType.HEADER}")

    print("\n  Layer map:")
    for name, num in [("Gate", Layer.GATE), ("Metal1", Layer.METAL1),
                      ("Metal2", Layer.METAL2), ("Via1", Layer.VIA1),
                      ("Active", Layer.ACTIVE), ("Pad", Layer.PAD)]:
        print(f"    {name:10s}: Layer {num}")


if __name__ == "__main__":
    demo()
