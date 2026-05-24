"""
Speculative Execution Manager
==============================
Detects straggler crawl tasks and re-issues them to idle workers.
The first result to arrive wins; the duplicate is silently discarded.

How it fits into the system
----------------------------
Master's TAG_READY handler calls find_straggler() before every normal
assignment. If a straggler is detected, the idle worker receives the
straggler's URL instead of a fresh one. When results arrive, handle_result()
decides whether to accept or discard — first arrival wins, regardless of
whether it was the original or speculative worker.

P95 baseline — why master-measured, not worker-reported
---------------------------------------------------------
worker.py already calls record_fetch(domain, duration_ms) in utils/metrics.py,
but that duration measures only the HTTP fetch. What matters for speculation
is total task round-trip: fetch + parse + MPI serialisation. We measure it
directly in master from in_progress start times, keeping this module
completely self-contained and avoiding changes to utils/metrics.py.

Fallback for new domains
--------------------------
When fewer than MIN_SAMPLES completions exist for a domain, we fall back to
the global p95 across all domains. If even that has too few samples, we skip
speculation for that task — conservative by design. A false speculation wastes
one worker's time; a missed speculation just means the straggler runs to
completion normally. The cost of false positives far exceeds missed detections.

PDC concept demonstrated
--------------------------
Speculative execution (also called redundant task assignment) was formalised
in distributed systems by the MapReduce paper's "Backup Tasks" mechanism
(Dean & Ghemawat, 2004, §3.6). The canonical result is that speculating
the slowest 5% of tasks reduces job completion time by up to 44% in practice.

Your system adds a domain-aware p95 threshold on top of the original
approach — the original MapReduce paper uses a global completion-fraction
heuristic ("task almost done → back it up"). Domain-awareness is more
precise: a Wikipedia fetch that runs 3× its p95 is a straggler; the same
elapsed time for a never-seen domain is normal. This is a meaningful
extension of the academic concept, not just an implementation of it.

Academic references
--------------------
Dean, J., & Ghemawat, S. (2004). MapReduce: Simplified Data Processing on
Large Clusters. OSDI'04. (§3.6 Backup Tasks)

Ananthanarayanan, G. et al. (2013). Effective Straggler Mitigation: Attack
of the Clones. NSDI'13. (domain-aware speculation rationale)
"""

import math
import time
import logging
from urllib.parse import urlparse

log = logging.getLogger("speculation")

# ── tunables ──────────────────────────────────────────────────────────────────

SPECULATION_FACTOR = 1.5   # speculate when elapsed > factor × p95(domain)
MIN_SAMPLES        = 5     # minimum completions before p95 is trustworthy
MAX_SAMPLE_SIZE    = 200   # rolling window — adapt to changing server latency


# ── helpers ───────────────────────────────────────────────────────────────────

def _domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
        return netloc if netloc else "unknown"
    except Exception:
        return "unknown"


def _percentile(samples: list, p: int) -> float:
    """
    Nearest-rank percentile over an unsorted sample list.
    Returns the value at the p-th percentile without interpolation —
    gives a conservative (higher) estimate, appropriate for a threshold.
    """
    sorted_s = sorted(samples)
    idx = math.ceil(p / 100.0 * len(sorted_s)) - 1
    return float(sorted_s[max(0, idx)])


# ── main class ────────────────────────────────────────────────────────────────

class SpeculationManager:
    """
    Tracks task timing, detects stragglers, and suppresses duplicate results.

    Lifecycle per speculated task
    ------------------------------
    1. record_completion() — called as normal tasks finish, builds p95 baseline
    2. find_straggler()    — called every TAG_READY; returns worst straggler or None
    3. record_speculation()— registered when master re-issues a straggler URL
    4. handle_result()     — called on every TAG_RESULT; returns accept/discard
    """

    def __init__(
        self,
        factor: float = SPECULATION_FACTOR,
        min_samples: int = MIN_SAMPLES,
    ):
        self.factor      = factor
        self.min_samples = min_samples

        # per-domain rolling sample windows  {domain: [duration_ms, ...]}
        self._samples: dict[str, list] = {}

        # doc_ids currently under speculative re-issue
        # {doc_id: {"orig_rank": int, "spec_rank": int, "issued_at": float}}
        self._speculated: dict[int, dict] = {}

        # doc_ids whose first result has already been accepted
        self._completed: set = set()

        # ── telemetry ────────────────────────────────────────────────────────
        self.total_speculated  = 0
        self.total_wins        = 0   # speculative copy arrived before original
        self.total_duplicates  = 0   # late results discarded
        self.spec_events: list = []  # full history for experiments/reports

    # ── step 1: build p95 baseline ────────────────────────────────────────────

    def record_completion(self, doc_id: int, domain: str, duration_ms: float):
        """
        Update the rolling p95 window for this domain.
        Call once per accepted result (not for discarded duplicates).

        duration_ms should be the master-measured round-trip time:
            (time.time() - in_progress[rank][3]) * 1000
        """
        if domain not in self._samples:
            self._samples[domain] = []
        samples = self._samples[domain]
        samples.append(duration_ms)
        # evict oldest sample to keep window bounded
        if len(samples) > MAX_SAMPLE_SIZE:
            samples.pop(0)

    # ── step 2: detect stragglers ─────────────────────────────────────────────

    def find_straggler(self, in_progress: dict, now: float):
        """
        Scan in_progress for the most overdue task and return it.

        Parameters
        ----------
        in_progress : {rank: (url, depth, doc_id, start_time)}
            The master's active task registry.
        now : float
            Current time.time() — passed in so caller controls the clock.

        Returns
        -------
        (orig_rank, url, depth, doc_id)  if a straggler is found
        None                              otherwise

        A task qualifies as a straggler when ALL of these hold:
          • elapsed_ms  >  SPECULATION_FACTOR × p95(domain)
          • p95 is based on >= MIN_SAMPLES completions (or global fallback)
          • doc_id has not already been speculatively re-issued
          • doc_id has not already produced an accepted result
        """
        worst_overdue   = 0.0
        worst_straggler = None

        for rank, task in in_progress.items():
            url, depth, doc_id, start_time = task

            if doc_id in self._speculated:
                continue    # already issued a speculative copy — don't double-speculate
            if doc_id in self._completed:
                continue    # result already in — master just hasn't processed it yet

            elapsed_ms = (now - start_time) * 1000
            domain     = _domain_of(url)
            p95        = self._get_p95(domain)

            if p95 is None:
                continue    # not enough data — skip rather than speculate blindly

            threshold_ms = self.factor * p95
            if elapsed_ms > threshold_ms:
                overdue = elapsed_ms - threshold_ms
                if overdue > worst_overdue:
                    worst_overdue   = overdue
                    worst_straggler = (rank, url, depth, doc_id)

        if worst_straggler is not None:
            _, url, _, doc_id = worst_straggler
            domain = _domain_of(url)
            log.info(
                f"[SPEC/DETECT] doc {doc_id} | domain '{domain}' "
                f"| overdue by {worst_overdue:.0f}ms "
                f"| p95={self._get_p95(domain):.0f}ms "
                f"| threshold={self.factor}×p95"
            )

        return worst_straggler

    # ── step 3: register the speculative re-issue ─────────────────────────────

    def record_speculation(self, doc_id: int, orig_rank: int, spec_rank: int):
        """
        Called immediately after master sends the speculative TAG_TASK.
        Locks in the mapping so future handle_result() calls know context.
        """
        entry = {
            "orig_rank": orig_rank,
            "spec_rank": spec_rank,
            "issued_at": time.time(),
        }
        self._speculated[doc_id] = entry
        self.total_speculated += 1
        self.spec_events.append({"doc_id": doc_id, **entry})
        log.info(
            f"[SPEC/ISSUE] doc {doc_id} re-issued to rank {spec_rank} "
            f"(original: rank {orig_rank})"
        )

    # ── step 4: accept or discard arriving results ────────────────────────────

    def handle_result(self, doc_id: int, arriving_rank: int) -> bool:
        """
        Decide whether to accept or discard this result.

        Returns
        -------
        True  — first arrival, accept and save to DB as normal
        False — duplicate, discard silently

        Call this before any save_page() or link-enqueueing logic.
        Only call record_completion() when this returns True.
        """
        if doc_id in self._completed:
            # ── duplicate ─────────────────────────────────────────────────────
            self.total_duplicates += 1
            spec_info = self._speculated.get(doc_id, {})
            log.info(
                f"[SPEC/DISCARD] doc {doc_id} from rank {arriving_rank} "
                f"— late duplicate discarded "
                f"(orig: rank {spec_info.get('orig_rank', '?')}, "
                f"spec: rank {spec_info.get('spec_rank', '?')})"
            )
            return False

        # ── first arrival: accept ─────────────────────────────────────────────
        self._completed.add(doc_id)

        if doc_id in self._speculated:
            spec_info = self._speculated[doc_id]
            latency_saved_ms = (time.time() - spec_info["issued_at"]) * 1000

            if arriving_rank == spec_info["spec_rank"]:
                # speculative copy won the race
                self.total_wins += 1
                log.info(
                    f"[SPEC/WIN] doc {doc_id}: speculative rank {arriving_rank} "
                    f"beat original rank {spec_info['orig_rank']} "
                    f"(~{latency_saved_ms:.0f}ms saved)"
                )
            else:
                # original completed — speculative copy will arrive later and
                # be discarded by the duplicate check above
                log.info(
                    f"[SPEC/ORIG] doc {doc_id}: original rank {arriving_rank} "
                    f"completed normally; speculative rank {spec_info['spec_rank']} "
                    f"result will be discarded when it arrives"
                )

        return True

    # ── introspection ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        domain_p95s = {
            domain: round(_percentile(samples, 95), 1)
            for domain, samples in self._samples.items()
            if len(samples) >= self.min_samples
        }
        return {
            "total_speculated":   self.total_speculated,
            "total_wins":         self.total_wins,
            "total_duplicates":   self.total_duplicates,
            "win_rate":           round(
                self.total_wins / max(self.total_speculated, 1), 3
            ),
            "domain_p95_ms":      domain_p95s,
            "samples_per_domain": {d: len(s) for d, s in self._samples.items()},
        }

    def is_speculative_doc(self, doc_id: int) -> bool:
        return doc_id in self._speculated

    # ── p95 helpers ───────────────────────────────────────────────────────────

    def _get_p95(self, domain: str):
        """
        Return p95 for this domain, falling back to global p95 if the
        domain has fewer than MIN_SAMPLES. Returns None when even the
        global pool has insufficient data.
        """
        samples = self._samples.get(domain, [])
        if len(samples) >= self.min_samples:
            return _percentile(samples, 95)

        # global fallback: all recorded durations across all domains
        all_samples = [s for v in self._samples.values() for s in v]
        if len(all_samples) >= self.min_samples:
            return _percentile(all_samples, 95)

        return None   # too early in the crawl to speculate reliably
