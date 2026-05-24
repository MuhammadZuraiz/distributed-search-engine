"""
Push-Sum Epidemic Aggregator
==============================
Computes distributed aggregates (sum, mean) via gossip without
any collective barrier operation.

The problem this solves
------------------------
The MapReduce indexer calls comm.gather() to collect all term/doc pairs
at rank 0. Before rank 0 can build BM25 scores it needs two global stats:

    total_docs  — sum of local_doc_count across all mappers
    avgdl       — sum of local_total_length / total_docs

In the current code those stats are computed AFTER the gather completes,
meaning the sequential reduce phase has two stages: (1) wait for all
mappers, (2) compute stats, (3) build index. Push-sum eliminates stage 2
by computing the stats IN PARALLEL with the map phase, so they are ready
the moment the gather completes.

The broader architectural implication is that every mapper ends the map
phase already knowing the global corpus statistics. That is what makes
distributed BM25 scoring possible — each shard can score independently
without a coordinator.

Algorithm: Push-Sum (Kempe, Dobra, Gehrke — FOCS 2003)
---------------------------------------------------------
State per node: a pair (v_k, w_k) for each aggregate k.
  v_k  = value estimate
  w_k  = weight (starts at 1.0)

Each round:
  1. Halve own (v, w): keep half, prepare half to send
  2. Send half to a deterministic peer
  3. Receive half from a deterministic peer
  4. Add received values to own halves

Invariant: sum(v_k) across all nodes is preserved each round.
           sum(w_k) across all nodes is preserved each round.

Convergence: after R rounds, v_k / w_k → global_mean(initial v_k) for
all nodes. Convergence is exponential: error halves each round.

Global sum  = v_k / w_k * n       (mean × count)
Global mean = v_k / w_k

Peer selection: ring-doubling (stride = 2^r)
---------------------------------------------
In round r, node i sends to (i + 2^r) % n and receives from
(i - 2^r + n) % n. This ensures:
  - Every node i has a unique send-target and receive-source each round
  - The communication graph covers all nodes in ceil(log2(n)) rounds
  - No deadlock: all sends are non-blocking (isend), receives are blocking

Deadlock proof sketch
----------------------
Every node calls isend (non-blocking, returns immediately) before
calling recv (blocking). isend posts the send buffer to MPI and
returns — it does not wait for the peer to recv. Therefore no node
blocks before its message is available to its peer's recv. Since
every recv has a corresponding isend already posted by the time recv
is called, all recvs complete. QED.

Academic reference
-------------------
Kempe, D., Dobra, A., & Gehrke, J. (2003). Gossip-based computation
of aggregate information. FOCS 2003.
"""

import math
import time
import logging

log = logging.getLogger("push_sum")

# MPI tag reserved for push-sum messages.
# Must not collide with any tag used in master.py / worker.py.
# Those use tags 1–9; we use 50–59 here.
_BASE_TAG = 50


class PushSumAggregator:
    """
    Runs push-sum gossip to compute global sums and means across all MPI ranks.

    Usage
    ------
        # Each rank initialises with its LOCAL values
        agg = PushSumAggregator(comm, {
            "doc_count":    float(local_doc_count),
            "total_length": float(local_total_length),
        })

        # Run gossip — returns converged estimates on every rank
        estimates = agg.run()

        n           = comm.Get_size()
        total_docs  = round(estimates["doc_count"]    * n)
        total_len   = round(estimates["total_length"] * n)
        avgdl       = total_len / max(total_docs, 1)

    Parameters
    ----------
    comm       : MPI.Comm
    init_vals  : dict[str, float]   local initial values per aggregate
    rounds     : int | None         gossip rounds (default: ceil(log2(n)) + 1)
    """

    def __init__(self, comm, init_vals: dict, rounds: int = None):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.keys = list(init_vals.keys())

        # push-sum state vectors
        self.v: dict[str, float] = {k: float(v) for k, v in init_vals.items()}
        self.w: dict[str, float] = {k: 1.0      for k in self.keys}

        # number of gossip rounds
        if rounds is not None:
            self.rounds = rounds
        else:
            self.rounds = math.ceil(math.log2(max(self.size, 2))) + 1

        # telemetry
        self._round_times: list[float] = []
        self._converged_at: int | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute all gossip rounds and return converged estimates.

        Returns
        -------
        dict[str, float]
            {key: global_mean} — multiply by comm.Get_size() to get global sum.

        All ranks must call run() — it is a collective in the sense that
        every rank participates, but it uses NO MPI collective operations
        (no Barrier, no Bcast, no Gather).
        """
        log.info(
            f"[push_sum] rank {self.rank} starting "
            f"{self.rounds} rounds over {self.size} nodes | "
            f"init: { {k: round(self.v[k], 2) for k in self.keys} }"
        )

        prev_estimates = self._estimates()

        for r in range(self.rounds):
            t0 = time.time()
            self._round(r)
            self._round_times.append(time.time() - t0)

            curr_estimates = self._estimates()

            # convergence check: stop early if max relative change < 0.1%
            if r >= 2:
                max_change = max(
                    abs(curr_estimates[k] - prev_estimates[k])
                    / max(abs(prev_estimates[k]), 1e-9)
                    for k in self.keys
                )
                if max_change < 0.001:
                    self._converged_at = r + 1
                    log.info(
                        f"[push_sum] rank {self.rank} converged "
                        f"after {r+1} rounds (change={max_change:.4%})"
                    )
                    break

            prev_estimates = curr_estimates

        final = self._estimates()
        log.info(
            f"[push_sum] rank {self.rank} done | "
            f"estimates: { {k: round(v, 4) for k, v in final.items()} } | "
            f"rounds_run: {len(self._round_times)}"
        )
        return final

    def global_sums(self) -> dict:
        """Convenience: return {key: estimated_global_sum} after run()."""
        estimates = self._estimates()
        return {k: estimates[k] * self.size for k in self.keys}

    def stats(self) -> dict:
        return {
            "rounds_planned":   self.rounds,
            "rounds_run":       len(self._round_times),
            "converged_at":     self._converged_at,
            "round_times_ms":   [round(t * 1000, 2) for t in self._round_times],
            "total_time_ms":    round(sum(self._round_times) * 1000, 2),
            "final_estimates":  self._estimates(),
        }

    # ── internal ──────────────────────────────────────────────────────────────

    def _round(self, r: int):
        """
        Execute one push-sum gossip round.

        Communication pattern: ring-doubling.
          send_to   = (rank + 2^r) % size
          recv_from = (rank - 2^r + size) % size
        """
        stride    = 1 << (r % math.ceil(math.log2(max(self.size, 2))))
        send_to   = (self.rank + stride) % self.size
        recv_from = (self.rank - stride + self.size) % self.size

        # halve own state — keep half, prepare send half
        send_v = {}
        send_w = {}
        for k in self.keys:
            half_v       = self.v[k] / 2.0
            half_w       = self.w[k] / 2.0
            self.v[k]    = half_v
            self.w[k]    = half_w
            send_v[k]    = half_v
            send_w[k]    = half_w

        # non-blocking send (isend): posts immediately, does not wait for peer
        tag = _BASE_TAG + (r % 10)
        req = self.comm.isend((send_v, send_w), dest=send_to, tag=tag)

        # blocking recv from peer: safe because peer has already posted isend
        recv_v, recv_w = self.comm.recv(source=recv_from, tag=tag)

        # wait for our own send to complete (buffer may now be freed)
        req.wait()

        # accumulate received halves
        for k in self.keys:
            self.v[k] += recv_v[k]
            self.w[k] += recv_w[k]

    def _estimates(self) -> dict:
        """Current estimate of global mean for each key: v_k / w_k."""
        return {k: self.v[k] / max(self.w[k], 1e-12) for k in self.keys}
