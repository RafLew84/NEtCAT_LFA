"""Fallback lattice definitions for GUI usage when analysis modules are unavailable."""

from __future__ import annotations

from typing import Dict

from ...core.constants import LATTICE_TYPE_HEXAGONAL, LATTICE_TYPE_SQUARE

KNOWN_LATTICES_FALLBACK: Dict[str, Dict] = {
    "Au(111)": {
        "type": LATTICE_TYPE_HEXAGONAL,
        "a_bulk": 0.408,  # nm
        "a_surf": 0.408 / 2 ** 0.5,
        "source": "Approx. bulk value",
    },
    "Ag(111)": {
        "type": LATTICE_TYPE_HEXAGONAL,
        "a_bulk": 0.409,
        "a_surf": 0.409 / 2 ** 0.5,
        "source": "Approx. bulk value",
    },
    "Cu(111)": {
        "type": LATTICE_TYPE_HEXAGONAL,
        "a_bulk": 0.361,
        "a_surf": 0.361 / 2 ** 0.5,
        "source": "Approx. bulk value",
    },
    "Pt(111)": {
        "type": LATTICE_TYPE_HEXAGONAL,
        "a_bulk": 0.392,
        "a_surf": 0.392 / 2 ** 0.5,
        "source": "Approx. bulk value",
    },
    "Ni(111)": {
        "type": LATTICE_TYPE_HEXAGONAL,
        "a_bulk": 0.352,
        "a_surf": 0.352 / 2 ** 0.5,
        "source": "Approx. bulk value",
    },
    "Graphene": {
        "type": LATTICE_TYPE_HEXAGONAL,
        "a_surf": 0.246,
        "source": "Typical value",
    },
    "HOPG": {
        "type": LATTICE_TYPE_HEXAGONAL,
        "a_surf": 0.246,
        "source": "Typical value",
    },
    "Au(100)": {
        "type": LATTICE_TYPE_SQUARE,
        "a_bulk": 0.408,
        "a_surf": 0.408 / 2 ** 0.5,
        "source": "Approx. bulk value",
    },
    "Ag(100)": {
        "type": LATTICE_TYPE_SQUARE,
        "a_bulk": 0.409,
        "a_surf": 0.409 / 2 ** 0.5,
        "source": "Approx. bulk value",
    },
    "Cu(100)": {
        "type": LATTICE_TYPE_SQUARE,
        "a_bulk": 0.361,
        "a_surf": 0.361 / 2 ** 0.5,
        "source": "Approx. bulk value",
    },
    "Pt(100)": {
        "type": LATTICE_TYPE_SQUARE,
        "a_bulk": 0.392,
        "a_surf": 0.392 / 2 ** 0.5,
        "source": "Approx. bulk value",
    },
    "Ni(100)": {
        "type": LATTICE_TYPE_SQUARE,
        "a_bulk": 0.352,
        "a_surf": 0.352 / 2 ** 0.5,
        "source": "Approx. bulk value",
    },
}

__all__ = ["KNOWN_LATTICES_FALLBACK"]
