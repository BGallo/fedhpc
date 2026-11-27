"""List scheduling on interval structure: a primal heuristic for
time-indexed models, read from the matrix alone.

Structure (``detect_interval_structure``): assignment rows (all
coefficients 1, over binaries, requiring at least one: ``sum x >= 1`` or
``= 1``) that partition the binary columns into jobs, and an interval row
block (``reformulate.detect_interval_blocks``) in which each column's
entries are one run [a, e] of consecutive rows with a constant coefficient
v: the column starts its job at block row a, occupies it until e and uses v
of each row's capacity ``row_hi``. Time-indexed scheduling (Sousa & Wolsey
1992; Kramer et al.'s TI) is exactly this: x_js covers periods s..s+p_j-1 of
the machine-count rows.

Schedules (``schedule``): jobs in a priority order each take their cheapest
column whose run fits the residual capacity (first fit; with costs rising
in start time, the earliest start), which is Graham's list scheduling with
release dates when each run is a job on one of row_hi machines. A window
fits iff no row in it lacks capacity: one prefix sum over the block per job
answers it for every candidate column at once.

Orders:
- Smith's ratio p_j / w_j (WSPT, Smith 1956), with p_j a job's run length
  and w_j the least-squares slope of its columns' costs over their start
  rows (the weight of a completion-time objective), for a schedule before
  any LP;
- LP start times, mean and alpha-points (Hall, Schulz, Shmoys & Wein 1997;
  Savelsbergh, Uma & Wein 2005), after the root LP;
- then insertion local search on the best order, within a time budget.

Arc-flow models (``detect_flow_structure``, ``flow_schedule``): equality
rows with +-1 entries form a DAG (each column one +1 at its tail row, one
-1 at its head; positive right-hand sides supply machines, negative ones
absorb them), job arcs are the binaries of assignment rows and the other
arcs are idle ("loss") arcs. Machines are tokens walked through the nodes
in topological order (ties by row index, the builders' time order): at its
node a machine takes the unassigned job of best priority with an arc
leaving there, else the idle arc to the earliest node - list scheduling
read off the network (Kramer et al.'s AF; Valerio de Carvalho's arc flow).

A schedule is returned only when it is feasible for the whole problem
(other rows included).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .problem import MipProblem


@dataclass
class IntervalStructure:
    job_cols: list[np.ndarray]      # per job: its columns with a run, sorted by start
    start: np.ndarray               # per column: block row where its run starts (-1: none)
    end: np.ndarray                 # last row of the run (inclusive)
    coef: np.ndarray                # the run's coefficient
    blk_lo: int                     # the block rows' range [blk_lo, blk_hi]
    blk_hi: int
    cap: np.ndarray                 # row_hi over the block rows
    p: np.ndarray                   # per job: mean run length
    w: np.ndarray                   # per job: cost slope over start (weight)


def detect_interval_structure(P: MipProblem, blocks) -> IntervalStructure | None:
    """Jobs and runs over one interval block, or None. ``blocks``: the
    detected interval blocks (the first whose rows carry upper capacities
    and cover the jobs' columns is used)."""
    if not blocks:
        return None
    A = P.A
    binary = P.integer & (P.lb == 0) & (P.ub == 1)
    rn = np.diff(A.indptr)
    rows = np.arange(P.m)
    # rows with all coefficients 1 (rows' max == min == 1)
    if A.nnz == 0:
        return None
    rr = np.repeat(rows, rn)
    one = np.ones(P.m, bool)
    np.logical_and.at(one, rr, A.data == 1.0)
    allbin = np.ones(P.m, bool)
    np.logical_and.at(allbin, rr, binary[A.indices])
    for (lo, hi) in blocks:
        in_blk = np.zeros(P.m, bool)
        in_blk[lo:hi + 1] = True
        if not np.all(np.isfinite(P.row_hi[lo:hi + 1])):
            continue
        assign = ~in_blk & one & allbin & (rn > 0) & (P.row_lo >= 1 - 1e-9) & (P.row_hi >= 1)
        if not np.any(assign):
            continue
        # every column in at most one assignment row
        sub = A[np.nonzero(assign)[0]]
        per_col = np.bincount(sub.indices, minlength=P.n)
        if np.any(per_col > 1):
            continue
        # runs in the block
        start = np.full(P.n, -1, np.int64)
        end = np.full(P.n, -1, np.int64)
        coef = np.zeros(P.n)
        blk = A[lo:hi + 1].tocsc()
        blk.sort_indices()
        nz = np.diff(blk.indptr)
        has = nz > 0
        first = blk.indptr[:-1]
        last = blk.indptr[1:] - 1
        cols = np.nonzero(has)[0]
        s_row = blk.indices[first[cols]]
        e_row = blk.indices[last[cols]]
        v = blk.data[first[cols]]
        # contiguity and constant coefficient were established by detection;
        # check the run length agrees with the entry count
        ok = (e_row - s_row + 1) == nz[cols]
        if not np.all(ok):
            continue
        start[cols], end[cols], coef[cols] = s_row + lo, e_row + lo, v
        if np.any(coef[cols] <= 0):
            continue
        job_cols, ps, ws = [], [], []
        jr = np.nonzero(assign)[0]
        for r in jr:
            cj = A.indices[A.indptr[r]:A.indptr[r + 1]]
            cj = cj[start[cj] >= 0]
            if len(cj) == 0:
                break
            cj = cj[np.lexsort((P.c[cj], start[cj]))]
            job_cols.append(cj)
            ps.append(float(np.mean(end[cj] - start[cj] + 1)))
            st = start[cj].astype(float)
            if len(cj) > 1 and np.ptp(st) > 0:
                ws.append(float(np.polyfit(st, P.c[cj], 1)[0]))
            else:
                ws.append(0.0)
        else:
            return IntervalStructure(job_cols, start, end, coef, lo, hi, P.row_hi[lo:hi + 1].astype(float).copy(),
                                     np.asarray(ps), np.asarray(ws))
    return None


def schedule(P: MipProblem, S: IntervalStructure, order, tol: float = 1e-9) -> np.ndarray | None:
    """List-schedule the jobs in ``order``; the 0/1 point, or None when a
    job finds no column that fits."""
    cap = S.cap.copy()
    base = S.blk_lo
    x = np.zeros(P.n)
    for k in order:
        cj = S.job_cols[k]
        a, e, v = S.start[cj] - base, S.end[cj] - base, S.coef[cj]
        fits = np.zeros(len(cj), bool)
        for vv in np.unique(v):
            short = np.concatenate(([0], np.cumsum(cap < vv - tol)))
            sel = v == vv
            fits[sel] = short[e[sel] + 1] - short[a[sel]] == 0
        if not np.any(fits):
            return None
        idx = np.nonzero(fits)[0]
        best = idx[np.argmin(P.c[cj[idx]])]     # cheapest fitting (ties: earliest, by the sort)
        j = cj[best]
        x[j] = 1.0
        cap[a[best]:e[best] + 1] -= v[best]
    return x


def orders_before_lp(S: IntervalStructure) -> list[np.ndarray]:
    n = len(S.job_cols)
    w = np.where(S.w > 0, S.w, 1e-12)
    first_start = np.array([S.start[c[0]] for c in S.job_cols], float)
    smith = np.lexsort((first_start, S.p / w))
    release = np.lexsort((S.p / w, first_start))
    return [smith, release] if n > 1 else [np.arange(n)]


def orders_from_lp(S: IntervalStructure, x: np.ndarray, alphas=(0.5, 0.2, 0.8)) -> list[np.ndarray]:
    """Jobs by LP mean start, then by alpha-points: the first start at which
    a job's cumulative LP mass reaches alpha."""
    out = []
    w = np.where(S.w > 0, S.w, 1e-12)
    smith = S.p / w
    means, apts = [], {a: [] for a in alphas}
    for cj in S.job_cols:
        xv = np.clip(x[cj], 0, None)
        tot = xv.sum()
        st = S.start[cj].astype(float)
        means.append(float(xv @ st / tot) if tot > 0 else float(st[0]))
        cum = np.cumsum(xv) / tot if tot > 0 else np.ones(len(cj))
        for a in alphas:
            apts[a].append(float(st[min(np.searchsorted(cum, a - 1e-12), len(cj) - 1)]))
    out.append(np.lexsort((smith, np.asarray(means))))
    for a in alphas:
        out.append(np.lexsort((np.asarray(means), np.asarray(apts[a]))))
    return out


def best_schedule(P: MipProblem, S, orders, deadline: float, improve: bool = True,
                  max_evals: int = 3000):
    """The best feasible schedule over the orders, then insertion local
    search (move one job earlier or later in the order) on the best order
    until no move improves, ``max_evals`` schedules have been built (the
    deterministic bound) or the deadline passes. Returns (x, objective)."""
    run = flow_schedule if isinstance(S, FlowStructure) else schedule
    best_x, best_v, best_o = None, np.inf, None
    for o in orders:
        x = run(P, S, o)
        if x is None or not P.is_feasible(x):
            continue
        v = float(P.c @ x)
        if v < best_v:
            best_x, best_v, best_o = x, v, list(o)
    if best_x is None or not improve:
        return best_x, best_v
    n = len(best_o)
    improved = True
    evals = 0
    while improved and time.perf_counter() < deadline and evals < max_evals:
        improved = False
        for i in range(n):
            if time.perf_counter() > deadline or evals >= max_evals:
                break
            for d in (-1, 1, -3, 3):
                k = i + d
                if not 0 <= k < n:
                    continue
                o = best_o.copy()
                job = o.pop(i)
                o.insert(k, job)
                evals += 1
                x = run(P, S, o)
                if x is None:
                    continue
                v = float(P.c @ x)
                if v < best_v - 1e-9 and P.is_feasible(x):
                    best_x, best_v, best_o = x, v, o
                    improved = True
                    break
    return best_x, best_v


# ------------------------------------------------------------- arc flow
@dataclass
class FlowStructure:
    job_cols: list[np.ndarray]      # per job: its arcs
    job_of: np.ndarray              # per column: its job (-1: an idle arc / not a flow arc)
    tail: np.ndarray                # per column: tail node (rank), -1 when not a flow arc
    head: np.ndarray
    supply: np.ndarray              # per node rank: tokens starting there
    out_jobs: list                  # per node rank: job arcs leaving it
    out_idle: list                  # per node rank: idle arcs leaving it, by head rank
    p: np.ndarray                   # per job: mean arc length in ranks (for orders before LP)
    w: np.ndarray


def detect_flow_structure(P: MipProblem) -> FlowStructure | None:
    """Arc-flow structure (see the module notes), or None."""
    A = P.A
    if P.m == 0 or A.nnz == 0:
        return None
    rn = np.diff(A.indptr)
    rr = np.repeat(np.arange(P.m), rn)
    binary = P.integer & (P.lb == 0) & (P.ub == 1)
    one = np.ones(P.m, bool)
    np.logical_and.at(one, rr, A.data == 1.0)
    allbin = np.ones(P.m, bool)
    np.logical_and.at(allbin, rr, binary[A.indices])
    unit = np.ones(P.m, bool)
    np.logical_and.at(unit, rr, np.abs(A.data) == 1.0)
    with np.errstate(invalid="ignore"):
        eq = (P.row_lo == P.row_hi) & np.isfinite(P.row_hi) & (np.abs(P.row_hi - np.round(P.row_hi)) < 1e-9)
    assign = one & allbin & (rn > 0) & (P.row_lo >= 1 - 1e-9) & (P.row_hi >= 1) & ~eq
    flow = eq & unit & ~assign
    if not np.any(flow) or not np.any(assign):
        return None
    sub = A[np.nonzero(assign)[0]]
    if np.any(np.bincount(sub.indices, minlength=P.n) > 1):
        return None
    job_of = np.full(P.n, -1, np.int64)
    jr = np.nonzero(assign)[0]
    for k, r in enumerate(jr):
        job_of[A.indices[A.indptr[r]:A.indptr[r + 1]]] = k
    # each column's flow entries: one +1 (tail) and one -1 (head)
    F = A[np.nonzero(flow)[0]].tocsc()
    frow = np.nonzero(flow)[0]
    cnt = np.diff(F.indptr)
    tail = np.full(P.n, -1, np.int64)
    head = np.full(P.n, -1, np.int64)
    cols = np.nonzero(cnt > 0)[0]
    if np.any(cnt[cols] != 2):
        return None
    a0, a1 = F.data[F.indptr[cols]], F.data[F.indptr[cols] + 1]
    r0, r1 = F.indices[F.indptr[cols]], F.indices[F.indptr[cols] + 1]
    if np.any(a0 + a1 != 0):
        return None
    tail[cols] = np.where(a0 > 0, r0, r1)
    head[cols] = np.where(a0 > 0, r1, r0)
    if np.any((job_of >= 0) & (tail < 0)):
        return None                                    # a job column off the network
    idle = (job_of < 0) & (tail >= 0)
    if np.any(idle & P.integer & (P.ub < 1)):
        return None
    # topological order (Kahn), ties by row index
    nf = len(frow)
    indeg = np.bincount(head[cols], minlength=nf)
    succ = [[] for _ in range(nf)]
    for c in cols.tolist():
        succ[tail[c]].append(head[c])
    import heapq
    ready = [v for v in range(nf) if indeg[v] == 0]
    heapq.heapify(ready)
    rank = np.full(nf, -1, np.int64)
    k = 0
    while ready:
        v = heapq.heappop(ready)
        rank[v] = k
        k += 1
        for u in succ[v]:
            indeg[u] -= 1
            if indeg[u] == 0:
                heapq.heappush(ready, u)
    if k < nf:
        return None                                    # a cycle: not a time network
    tail[cols], head[cols] = rank[tail[cols]], rank[head[cols]]
    supply = np.zeros(nf)
    supply[rank] = P.row_hi[frow]
    if np.any(np.abs(supply - np.round(supply)) > 1e-9):
        return None
    out_jobs = [[] for _ in range(nf)]
    out_idle = [[] for _ in range(nf)]
    for c in cols.tolist():
        (out_jobs if job_of[c] >= 0 else out_idle)[tail[c]].append(c)
    for v in range(nf):
        out_idle[v].sort(key=lambda c: head[c])
    job_cols = [np.nonzero(job_of == k)[0] for k in range(len(jr))]
    if any(len(c) == 0 for c in job_cols):
        return None
    ps, ws = [], []
    for cj in job_cols:
        ln = (head[cj] - tail[cj]).astype(float)
        ps.append(float(ln.mean()))
        st = tail[cj].astype(float)
        ws.append(float(np.polyfit(st, P.c[cj], 1)[0]) if len(cj) > 1 and np.ptp(st) > 0 else 0.0)
    return FlowStructure(job_cols, job_of, tail, head, np.round(supply), out_jobs, out_idle,
                         np.asarray(ps), np.asarray(ws))


def flow_schedule(P: MipProblem, S: FlowStructure, order) -> np.ndarray | None:
    """The better of the serial and the non-delay schedule of an order."""
    best, bv = None, np.inf
    for fn in (flow_schedule_serial, flow_schedule_nondelay):
        x = fn(P, S, order)
        if x is not None:
            v = float(P.c @ x)
            if v < bv:
                best, bv = x, v
    return best


def flow_schedule_serial(P: MipProblem, S: FlowStructure, order) -> np.ndarray | None:
    """Serial schedule generation: jobs in order, each appended to a
    machine at the cheapest of its arcs whose tail the machine reaches from
    its current end through idle arcs (the earliest idle arc out of each
    node: the chain to the next release); among machines reaching it, the
    one that idles least."""
    nf = len(S.supply)
    nxt = np.full(nf, -1, np.int64)
    for v in range(nf):
        if S.out_idle[v]:
            nxt[v] = S.head[S.out_idle[v][0]]
    ends = []
    for v in np.nonzero(S.supply > 0)[0].tolist():
        ends += [v] * int(S.supply[v])
    x = np.zeros(P.n)

    def chain(u):
        out = {}
        k = 0
        while u >= 0 and u not in out:
            out[u] = k
            u = int(nxt[u])
            k += 1
        return out

    reach = [chain(u) for u in ends]
    for j in order:
        cj = S.job_cols[j]
        best, bc, bm = -1, np.inf, -1
        for c in cj[np.argsort(P.c[cj], kind="stable")].tolist():
            if P.c[c] > bc:
                break
            t = int(S.tail[c])
            pick, steps = -1, None
            for i, R in enumerate(reach):
                k = R.get(t)
                if k is not None and (steps is None or k < steps):
                    pick, steps = i, k
            if pick >= 0:
                best, bc, bm = c, P.c[c], pick
                break
        if best < 0:
            return None
        u, t = ends[bm], int(S.tail[best])
        while u != t:                                   # idle along the chain
            a = S.out_idle[u][0]
            x[a] += 1.0
            u = int(S.head[a])
        x[best] += 1.0
        ends[bm] = int(S.head[best])
        reach[bm] = chain(ends[bm])
    for u in ends:                                      # idle to the sink
        while S.out_idle[u]:
            a = S.out_idle[u][0]
            x[a] += 1.0
            u = int(S.head[a])
        if S.supply[u] >= 0 and S.out_jobs[u]:
            return None
    if np.any(x > P.ub + 1e-9):
        return None
    return x


def flow_schedule_nondelay(P: MipProblem, S: FlowStructure, order) -> np.ndarray | None:
    """Walk the machines through the network in node order; a machine at a
    node takes the unassigned job of best priority with an arc leaving it,
    else the idle arc to the earliest node."""
    nj = len(S.job_cols)
    prio = np.empty(nj, np.int64)
    prio[np.asarray(order)] = np.arange(nj)
    tokens = S.supply.clip(min=0).astype(np.int64)
    done = np.zeros(nj, bool)
    x = np.zeros(P.n)
    nf = len(tokens)
    for v in range(nf):
        k = int(tokens[v])
        if k <= 0:
            continue
        for _ in range(k):
            best, bp = -1, nj
            for c in S.out_jobs[v]:
                j = S.job_of[c]
                if not done[j] and prio[j] < bp:
                    best, bp = c, prio[j]
            if best < 0:
                if not S.out_idle[v]:
                    break                               # a sink (or a dead end)
                best = S.out_idle[v][0]
            else:
                done[S.job_of[best]] = True
            x[best] += 1.0
            tokens[S.head[best]] += 1
            tokens[v] -= 1
    if not np.all(done):
        return None
    if np.any(x > P.ub + 1e-9):
        return None
    return x


def flow_orders(S: FlowStructure, x: np.ndarray | None) -> list[np.ndarray]:
    """Job priorities: LP mean tail rank (and alpha-points) with an LP
    point; Smith's ratio over rank lengths and cost slopes, and release
    order, always."""
    nj = len(S.job_cols)
    w = np.where(S.w > 0, S.w, 1e-12)
    smith = S.p / w
    first = np.array([S.tail[c].min() for c in S.job_cols], float)
    out = [np.lexsort((first, smith)), np.lexsort((smith, first))]
    if x is not None:
        means, a5 = [], []
        for cj in S.job_cols:
            xv = np.clip(x[cj], 0, None)
            o = np.argsort(S.tail[cj], kind="stable")
            st, xv = S.tail[cj][o].astype(float), xv[o]
            tot = xv.sum()
            means.append(float(xv @ st / tot) if tot > 0 else float(st[0]))
            cum = np.cumsum(xv) / tot if tot > 0 else np.ones(len(st))
            a5.append(float(st[min(np.searchsorted(cum, 0.5 - 1e-12), len(st) - 1)]))
        out = [np.lexsort((smith, np.asarray(means))), np.lexsort((np.asarray(means), np.asarray(a5)))] + out
    return out if nj > 1 else [np.arange(nj)]


def detect(P: MipProblem, blocks) -> IntervalStructure | FlowStructure | None:
    """The interval structure over the given blocks, else arc-flow."""
    S = detect_interval_structure(P, blocks) if blocks else None
    return S if S is not None else detect_flow_structure(P)


def orders(S, x: np.ndarray | None = None) -> list[np.ndarray]:
    """Priority orders: from the LP point x when given, plus the LP-free ones."""
    if isinstance(S, FlowStructure):
        return flow_orders(S, x)
    return (orders_from_lp(S, x) if x is not None else []) + orders_before_lp(S)
