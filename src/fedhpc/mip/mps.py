"""MPS reader (fixed and free format, optionally gzip-compressed).

Sections: NAME, OBJSENSE (MIN/MAX, inline or on the next line), ROWS, COLUMNS
(with MARKER INTORG/INTEND blocks), RHS, RANGES, BOUNDS, ENDATA. The first N
row is the objective; further N rows are dropped. An RHS entry on the
objective row is minus the objective's constant. Fields are whitespace
separated, so names must not contain spaces (true of MIPLIB 2017).

Conventions, as CPLEX, Gurobi and HiGHS read them:
- RANGES on row r with value R: an L row gets [rhs - |R|, rhs], a G row
  [rhs, rhs + |R|], an E row [rhs, rhs + R] for R > 0 and [rhs + R, rhs] for
  R < 0.
- Integer columns (in INTORG blocks) without bounds are [0, inf).
- UP with a negative value on a column whose lower bound is 0 makes the
  lower bound -inf.
- Bound types UP LO FX FR MI PL BV LI UI. Semi-continuous (SC) columns are
  rejected.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .problem import INF, MipProblem


def read_mps(path: str | Path) -> MipProblem:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return parse_mps(f, default_name=path.name.split(".")[0])


def parse_mps(lines, default_name: str = "") -> MipProblem:
    name = default_name
    maximize = False
    section = None
    obj_row = None
    row_index: dict[str, int] = {}
    row_sense: list[str] = []
    row_names: list[str] = []
    drop_rows: set[str] = set()
    col_index: dict[str, int] = {}
    col_names: list[str] = []
    is_int: list[bool] = []
    obj: list[float] = []
    ri: list[int] = []
    ci: list[int] = []
    vals: list[float] = []
    rhs: dict[int, float] = {}
    obj_rhs = 0.0
    ranges: dict[int, float] = {}
    bounds: list[tuple[str, int, float]] = []
    integer_block = False

    def col(cname: str) -> int:
        j = col_index.get(cname)
        if j is None:
            j = len(col_names)
            col_index[cname] = j
            col_names.append(cname)
            is_int.append(integer_block)
            obj.append(0.0)
        return j

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("*"):
            continue
        if not line[0].isspace():   # a section header
            tok = line.split()
            head = tok[0].upper()
            if head == "NAME":
                name = tok[1] if len(tok) > 1 else name
                section = None
            elif head == "OBJSENSE":
                section = "OBJSENSE"
                if len(tok) > 1:
                    maximize = tok[1].upper() in ("MAX", "MAXIMIZE")
            elif head in ("ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS"):
                section = head
            elif head == "ENDATA":
                break
            elif head in ("OBJSENSE", "OBJSENCE"):
                section = "OBJSENSE"
            elif section == "OBJSENSE" and head in ("MAX", "MAXIMIZE", "MIN", "MINIMIZE"):
                maximize = head.startswith("MAX")
            else:
                raise ValueError(f"unknown MPS section {head!r}")
            continue
        tok = line.split()
        if section == "OBJSENSE":
            maximize = tok[0].upper() in ("MAX", "MAXIMIZE")
        elif section == "ROWS":
            kind, rname = tok[0].upper(), tok[1]
            if kind == "N":
                if obj_row is None:
                    obj_row = rname
                else:
                    drop_rows.add(rname)
                continue
            if kind not in ("L", "G", "E"):
                raise ValueError(f"unknown row type {kind!r}")
            row_index[rname] = len(row_names)
            row_names.append(rname)
            row_sense.append(kind)
        elif section == "COLUMNS":
            if len(tok) >= 3 and tok[1].strip("'").upper() == "MARKER":
                mk = tok[2].strip("'").upper()
                if mk == "INTORG":
                    integer_block = True
                elif mk == "INTEND":
                    integer_block = False
                continue
            j = col(tok[0])
            for k in range(1, len(tok) - 1, 2):
                rname, v = tok[k], float(tok[k + 1])
                if rname == obj_row:
                    obj[j] += v
                elif rname in drop_rows:
                    continue
                else:
                    r = row_index.get(rname)
                    if r is None:
                        raise ValueError(f"column {tok[0]} names unknown row {rname}")
                    ri.append(r), ci.append(j), vals.append(v)
        elif section == "RHS":
            # the set name is optional in free MPS: pairs start after it
            start = 1 if len(tok) % 2 == 1 else 0
            for k in range(start, len(tok) - 1, 2):
                rname, v = tok[k], float(tok[k + 1])
                if rname == obj_row:
                    obj_rhs = v
                elif rname in drop_rows:
                    continue
                else:
                    rhs[row_index[rname]] = v
        elif section == "RANGES":
            start = 1 if len(tok) % 2 == 1 else 0
            for k in range(start, len(tok) - 1, 2):
                if tok[k] in drop_rows or tok[k] == obj_row:
                    continue
                ranges[row_index[tok[k]]] = float(tok[k + 1])
        elif section == "BOUNDS":
            kind = tok[0].upper()
            if kind in ("FR", "MI", "PL", "BV"):
                cname = tok[2] if len(tok) >= 3 else tok[1]
                v = float(tok[3]) if (kind == "BV" and len(tok) >= 4) else 0.0
            else:
                if len(tok) == 4:
                    cname, v = tok[2], float(tok[3])
                elif len(tok) == 3:
                    cname, v = tok[1], float(tok[2])
                else:
                    raise ValueError(f"bad bound line: {line!r}")
            if kind == "SC":
                raise ValueError("semi-continuous columns are not supported")
            if cname not in col_index:
                raise ValueError(f"bound on unknown column {cname}")
            bounds.append((kind, col_index[cname], v))
        else:
            raise ValueError(f"data line outside a section: {line!r}")

    m, n = len(row_names), len(col_names)
    A = sp.csr_matrix((vals, (ri, ci)), shape=(m, n))
    row_lo = np.full(m, -INF)
    row_hi = np.full(m, INF)
    for r, s in enumerate(row_sense):
        b = rhs.get(r, 0.0)
        if s == "E":
            row_lo[r] = row_hi[r] = b
        elif s == "L":
            row_hi[r] = b
        else:
            row_lo[r] = b
        if r in ranges:
            R = ranges[r]
            if s == "L":
                row_lo[r] = b - abs(R)
            elif s == "G":
                row_hi[r] = b + abs(R)
            elif R > 0:
                row_hi[r] = b + R
            else:
                row_lo[r] = b + R
    integer = np.array(is_int, dtype=bool)
    lb = np.zeros(n)
    ub = np.full(n, INF)
    for kind, j, v in bounds:
        if kind == "UP":
            ub[j] = v
            if v < 0 and lb[j] == 0:
                lb[j] = -INF
        elif kind == "LO":
            lb[j] = v
        elif kind == "FX":
            lb[j] = ub[j] = v
        elif kind == "FR":
            lb[j], ub[j] = -INF, INF
        elif kind == "MI":
            lb[j] = -INF
        elif kind == "PL":
            ub[j] = INF
        elif kind == "BV":
            lb[j], ub[j] = 0.0, 1.0
            integer[j] = True
        elif kind == "LI":
            lb[j] = v
            integer[j] = True
        elif kind == "UI":
            ub[j] = v
            integer[j] = True
            if v < 0 and lb[j] == 0:
                lb[j] = -INF
        else:
            raise ValueError(f"unknown bound type {kind!r}")
    c = np.array(obj, dtype=float)
    offset = -obj_rhs
    if maximize:
        c, offset = -c, -offset
    return MipProblem(A, row_lo, row_hi, lb, ub, c, integer, offset=offset, name=name,
                      maximize=maximize, col_names=col_names, row_names=row_names)
