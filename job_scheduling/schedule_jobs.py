"""
priority_scheduler.py

Configurable scheduling engine supporting three priority strategies:
  - "ordered"  : schedule by descending priority per-type (greedy)
  - "weighted" : greedy across types, scoring candidate placements by priority*score
  - "optimal"  : CP-SAT optimization maximizing weighted scheduled minutes (requires ortools)

Auto-detects OR-Tools. If missing, optimal strategy will raise an informative error.

Usage:
  python priority_scheduler.py
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import json, csv, io
import math
import itertools
import statistics
import re

# plotting (optional, used by demo)
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Try to import OR-Tools
try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except Exception:
    HAS_ORTOOLS = False

# ---------------------------
# Data model & utilities
# ---------------------------
@dataclass
class Interval:
    start: int  # inclusive minutes since midnight
    end: int    # exclusive

    def duration(self) -> int:
        return max(0, self.end - self.start)

    def intersects(self, other: "Interval") -> bool:
        return not (self.end <= other.start or self.start >= other.end)

@dataclass
class Job:
    job_type: str
    start: int
    end: int
    priority: int = 1
    source: str = "existing"  # "existing" or "scheduled"

def fmt_time(mins: int) -> str:
    h = mins // 60
    m = mins % 60
    return f"{h:02d}:{m:02d}"

def merge_intervals(intervals: List[Interval]) -> List[Interval]:
    intervals = [iv for iv in intervals if iv.end > iv.start]
    if not intervals:
        return []
    intervals.sort(key=lambda iv: iv.start)
    merged = [Interval(intervals[0].start, intervals[0].end)]
    for iv in intervals[1:]:
        prev = merged[-1]
        if iv.start <= prev.end:
            prev.end = max(prev.end, iv.end)
        else:
            merged.append(Interval(iv.start, iv.end))
    return merged

def invert_intervals(intervals: List[Interval], horizon: Interval) -> List[Interval]:
    free = []
    prev_end = horizon.start
    for iv in intervals:
        if iv.start > prev_end:
            free.append(Interval(prev_end, iv.start))
        prev_end = max(prev_end, iv.end)
    if prev_end < horizon.end:
        free.append(Interval(prev_end, horizon.end))
    return free

# ---------------------------
# Scheduler core
# ---------------------------
class Scheduler:
    def __init__(self, jobs: List[Job], blackouts: List[Interval], gap: int, horizon: Interval):
        """
        jobs: list of existing jobs (Job.job_type, start, end). Some may have 'source'='scheduled' if added.
        blackouts: list of global no-run intervals
        gap: minimum gap (minutes) required after a job before another same-type job can start
        horizon: overall scheduling interval
        """
        self.jobs = list(jobs)
        self.blackouts = list(blackouts)
        self.gap = gap
        self.horizon = horizon

    def build_blocked_for_type(self, job_type: str, include_scheduled: bool=True) -> List[Interval]:
        blocked = []
        for j in self.jobs:
            if j.job_type == job_type:
                if not include_scheduled and j.source == "scheduled":
                    continue
                blocked.append(Interval(j.start - self.gap, j.end + self.gap))
        blocked.extend([Interval(b.start, b.end) for b in self.blackouts])
        # clamp to horizon
        for b in blocked:
            b.start = max(b.start, self.horizon.start)
            b.end = min(b.end, self.horizon.end)
        return merge_intervals(blocked)

    def feasible_second_run_windows(self, job_type: str, first_job_end: int, include_scheduled: bool=True) -> List[Tuple[Interval, int]]:
        """
        Returns list of (start_window [s,e), max_duration_if_start_at_s)
        start may be any minute t with s <= t < e; max duration when starting at s is (e - s).
        """
        blocked = self.build_blocked_for_type(job_type, include_scheduled=include_scheduled)
        free = invert_intervals(blocked, self.horizon)
        earliest_allowed = first_job_end + self.gap
        results = []
        for iv in free:
            s = max(iv.start, earliest_allowed)
            e = iv.end
            if e > s:
                results.append((Interval(s, e), e - s))
        return results

    def add_scheduled_job(self, job: Job):
        self.jobs.append(job)

    def remove_scheduled_jobs(self):
        self.jobs = [j for j in self.jobs if j.source != "scheduled"]

# ---------------------------
# Priority scheduling strategies
# ---------------------------
def schedule_priority_ordered(
    scheduler: 'Scheduler',
    first_end_map: dict,
    num_runs_per_type: dict,
    priority_map: dict,
    per_run_max_cap: int = None
) -> list:
    scheduled = []
    fe = dict(first_end_map)
    from collections import defaultdict
    existing_names = defaultdict(set)
    for j in scheduler.jobs:
        base = re.match(r"^([A-Z]+)", j.job_type).group(1) if re.match(r"^([A-Z]+)", j.job_type) else j.job_type
        existing_names[base].add(j.job_type)
    scheduled_names = defaultdict(set)
    # Read the original job names and priorities for scheduled jobs from jobs_to_schedule.csv
    import pandas as pd
    jobs_to_schedule_csv = None
    import inspect
    for frame in inspect.stack():
        if 'jobs_to_schedule_csv' in frame.frame.f_globals:
            jobs_to_schedule_csv = frame.frame.f_globals['jobs_to_schedule_csv']
            break
    if jobs_to_schedule_csv is None:
        jobs_to_schedule_csv = 'data/jobs_to_schedule.csv'
    df_sched = pd.read_csv(jobs_to_schedule_csv)

    # Schedule each job in jobs_to_schedule.csv individually, preserving job_name and priority
    scheduled_job_display_map = {}
    # Group jobs by type, preserving order
    jobs_by_type = {}
    for idx, row in df_sched.iterrows():
        t = row['job_type']
        if t not in jobs_by_type:
            jobs_by_type[t] = []
        jobs_by_type[t].append((idx, row))
    # Helper to check blackout overlap
    def is_in_blackout(start, end, blackouts):
        for b in blackouts:
            if not (end <= b.start or start >= b.end):
                return True
        return False

    # Schedule jobs per type independently
    for t in jobs_by_type:
        for job_idx, row in jobs_by_type[t]:
            job_name = row['job_name']
            priority = int(row['priority']) if 'priority' in row else priority_map.get(t, 1)
            base = re.match(r"^([A-Z]+)", t).group(1) if re.match(r"^([A-Z]+)", t) else t
            used = set(existing_names[base]) | set(scheduled_names[base])
            scheduled_flag = False
            day_offset = 0
            while not scheduled_flag:
                attempt_day = day_offset + 1
                print(f"[DEBUG] Attempting to schedule job {job_name} (type {t}) on day {attempt_day}", file=sys.stderr)
                # Use the full horizon, not just one day
                scheduler.horizon = Interval(0, scheduler.horizon.end)
                windows = scheduler.feasible_second_run_windows(t, fe[t], include_scheduled=True)
                win = None
                for w, maxdur in windows:
                    # ENFORCE: skip any window that overlaps with any blackout interval for its full duration
                    blocked = False
                    for b in scheduler.blackouts:
                        # If any part of the job's window overlaps with a blackout, block it
                        if not (w.end <= b.start or w.start >= b.end):
                            blocked = True
                            break
                    if blocked:
                        continue
                    # Accept the first available window anywhere in the horizon
                    win = (w, maxdur)
                    break
                if not win:
                    print(f"[DEBUG] No available window for {job_name} (type {t}) in the full horizon", file=sys.stderr)
                    break
                w, maxdur = win
                dur = maxdur if per_run_max_cap is None else min(maxdur, per_run_max_cap)
                if dur <= 0:
                    print(f"[DEBUG] Window found for {job_name} (type {t}) but duration {dur} <= 0", file=sys.stderr)
                    break
                start = w.start
                end = start + dur
                if start < 0:
                    print(f"[DEBUG] Negative start for {job_name} (type {t})", file=sys.stderr)
                    break
                # Schedule the job
                job = Job(t, start, end, int(row['priority']), 'scheduled')
                job.display_name = row['job_name']
                scheduled.append(job)
                scheduled_names[base].add(t)
                fe[t] = end
                scheduled_flag = True
                start = w.start
                end = start + dur
                if start < 0:
                    print(f"[DEBUG] Window found for {job_name} (type {t}) on day {attempt_day} but start < 0", file=sys.stderr)
                    day_offset += 1
                    continue
                print(f"[DEBUG] Scheduled {job_name} (type {t}) on day {attempt_day} at {fmt_time(start % 1440)}-{fmt_time(end % 1440)} (absolute mins {start}-{end})", file=sys.stderr)
                internal_name = f"{base}_sched_{job_idx+1}"
                display_name = job_name
                scheduled_job_display_map[internal_name] = display_name
                scheduled_names[base].add(internal_name)
                job = Job(t, start, end, priority=priority, source="scheduled")
                job.display_name = display_name
                scheduled.append(job)
                scheduler.add_scheduled_job(job)
                fe[t] = job.end
                scheduled_flag = True
    return {'jobs': scheduled, 'display_map': scheduled_job_display_map}

def schedule_priority_weighted(scheduler: Scheduler,
                               first_end_map: Dict[str,int],
                               num_runs_per_type: Dict[str,int],
                               priority_map: Dict[str,int],
                               per_run_max_cap: Optional[int]=None,
                               window_value_fn: Optional[Any]=None) -> List[Job]:
    """
    Weighted greedy:
      - At each step, consider the 'next' run for each job type that still needs runs.
      - Score candidate = priority[type] * window_value(window)
      - Choose the candidate with highest score, schedule it, update timeline, repeat.
    window_value_fn(window, maxdur, type) -> float; default uses maxdur (longer windows preferred).
    """
    if window_value_fn is None:
        window_value_fn = lambda window, maxdur, t: maxdur  # prefer longer windows

    scheduled = []
    remaining = dict(num_runs_per_type)
    fe_map = dict(first_end_map)

    # Load job_name mapping from jobs_to_schedule.csv
    import pandas as pd
    try:
        df_sched = pd.read_csv('data/jobs_to_schedule.csv')
        # Build a mapping: job_type -> list of job_names (in order)
        job_name_map = {}
        for t in df_sched['job_type'].unique():
            job_name_map[t] = list(df_sched[df_sched['job_type'] == t]['job_name'])
        job_name_idx = {t: 0 for t in job_name_map}
    except Exception:
        job_name_map = {}
        job_name_idx = {}

    # Continue until all runs placed or no candidate windows
    while any(remaining.get(t,0) > 0 for t in remaining):
        candidates = []
        for t, rem in remaining.items():
            if rem <= 0:
                continue
            windows = scheduler.feasible_second_run_windows(t, fe_map[t], include_scheduled=True)
            if not windows:
                continue
            # For scoring, consider best window for this type (e.g., earliest or maxdur)
            # We'll compute best window_value among windows to represent candidate
            best = max(windows, key=lambda w: window_value_fn(w[0], w[1], t))
            win, maxdur = best
            score = priority_map.get(t,1) * window_value_fn(win, maxdur, t)
            candidates.append((score, t, win, maxdur))
        if not candidates:
            break
        # pick highest-score candidate; tie-breaker: higher priority, then type lexicographic
        candidates.sort(key=lambda x: (-x[0], -priority_map.get(x[1],1), x[1]))
        _, chosen_t, chosen_win, chosen_maxdur = candidates[0]
        dur = chosen_maxdur if per_run_max_cap is None else min(chosen_maxdur, per_run_max_cap)
        if dur <= 0:
            remaining[chosen_t] -= 1
            continue
        start = chosen_win.start
        end = start + dur
        # Enforce 24-hour constraint
        if start < 0 or end > 1440:
            remaining[chosen_t] -= 1
            continue
        job = Job(chosen_t, start, end, priority=priority_map.get(chosen_t,1), source="scheduled")
        # Assign display_name from job_name_map if available
        if chosen_t in job_name_map and job_name_idx[chosen_t] < len(job_name_map[chosen_t]):
            job.display_name = job_name_map[chosen_t][job_name_idx[chosen_t]]
            job_name_idx[chosen_t] += 1
        else:
            job.display_name = chosen_t
        scheduled.append(job)
        scheduler.add_scheduled_job(job)
        fe_map[chosen_t] = job.end
        remaining[chosen_t] -= 1
    return scheduled

# ---------------------------
# Optimal (CP-SAT) strategy - only if ortools available
# ---------------------------
def schedule_priority_optimal(scheduler: Scheduler,
                              first_end_map: Dict[str,int],
                              num_runs_per_type: Dict[str,int],
                              priority_map: Dict[str,int],
                              per_run_max_cap: Optional[int]=None,
                              time_granularity:int=1,
                              solver_time_limit_s: Optional[int]=30) -> List[Job]:
    """
    CP-SAT formulation:
      - For each requested run (type, idx), we create integer vars start_i and dur_i.
      - We precompute candidate windows for each run if needed and constrain start+dur <= window_end for chosen window via selection booleans (window selection).
      - Enforce non-overlap for runs of same type with mandatory gap.
      - Enforce blackout avoidance via requiring each run to be either before or after each blackout (using booleans).
      - Maximize sum(priority[type] * dur_i)
    Note: This is a fairly general model but can grow with number of windows and runs.
    """
    if not HAS_ORTOOLS:
        raise RuntimeError("OR-Tools not available. Install ortools to use the 'optimal' strategy (pip install ortools).")

    model = cp_model.CpModel()
    # prepare tasks list: flatten (type, run_index)
    tasks = []
    for t, n in num_runs_per_type.items():
        for i in range(n):
            tasks.append((t, i))
    if not tasks:
        return []

    # Precompute feasible windows per (type) using current scheduler state (existing jobs only)
    # We'll compute windows per type relative to the current first_end_map (but note the CP model ensures no-overlap)
    windows_per_type = {}
    for t in set(t for t,_ in tasks):
        windows_per_type[t] = scheduler.feasible_second_run_windows(t, first_end_map[t], include_scheduled=False)
        # windows_per_type[t] is list of (Interval, maxdur) where maxdur = interval.end - max(interval.start, earliest_allowed)
        # If no windows, we still continue (model may be infeasible)
    # Build variables per task
    task_vars = {}
    horizon_start = scheduler.horizon.start
    horizon_end = scheduler.horizon.end
    bigM = horizon_end - horizon_start + 1000

    for (t, i) in tasks:
        key = (t, i)
        # Start var domain: horizon (minute granularity), enforce 0 <= start < 1440
        s_var = model.NewIntVar(max(horizon_start, 0), min(horizon_end - 1, 1439), f"s_{t}_{i}")
        # Duration var domain: 0..max_possible (we allow 0 to represent "not scheduled" optionally, but we'll require >=1 by final constraints)
        # compute max window length for type
        max_len = 0
        for (win, maxdur) in windows_per_type.get(t, []):
            max_len = max(max_len, maxdur)
        if per_run_max_cap is not None:
            max_len = min(max_len, per_run_max_cap)
        if max_len <= 0:
            # force duration 0 and mark unschedulable by making objective ignore
            d_var = model.NewIntVar(0, 0, f"d_{t}_{i}")
        else:
            d_var = model.NewIntVar(1, max_len, f"d_{t}_{i}")
        # Enforce end <= 1440
        model.Add(s_var + d_var <= 1440)

        # We'll create a window-selection boolean for each window candidate for this type.
        wins = windows_per_type.get(t, [])
        win_bvars = []
        for widx, (win, maxdur) in enumerate(wins):
            b = model.NewBoolVar(f"sel_{t}_{i}_w{widx}")
            win_bvars.append((b, win))
            # If selected, start must be within [max(win.start, earliest_allowed), win.end - 1]
            earliest_allowed = max(win.start, first_end_map[t] + scheduler.gap)
            model.Add(s_var >= earliest_allowed).OnlyEnforceIf(b)
            model.Add(s_var <= win.end - 1).OnlyEnforceIf(b)
            # Enforce d_var <= win.end - s_var (i.e. must finish by window end) via big-M:
            # d_var + s_var <= win.end  <=> d_var <= win.end - s_var
            model.Add(d_var + s_var <= win.end).OnlyEnforceIf(b)

        if win_bvars:
            # exactly one window must be selected or zero? We want to allow unscheduling if impossible; but simpler: force exactly one selection
            model.Add(sum(b for b,_ in win_bvars) == 1)
        else:
            # no windows: force duration zero
            model.Add(d_var == 0)

        task_vars[key] = {"s": s_var, "d": d_var, "wins": win_bvars, "type": t}

    # No-overlap constraints for same-type runs (with gaps)
    for t, group in itertools.groupby(sorted(task_vars.items(), key=lambda kv: kv[0][0]), lambda kv: kv[0][0]):
        group = list(group)
        # group is iterable of ((type, i), vars)
        for a_idx in range(len(group)):
            for b_idx in range(a_idx+1, len(group)):
                (ta, ia), va = group[a_idx]
                (tb, ib), vb = group[b_idx]
                sa, da = va["s"], va["d"]
                sb, db = vb["s"], vb["d"]
                # Either a before b (sa + da + gap <= sb) OR b before a
                ba_before_b = model.NewBoolVar(f"{ta}_{ia}_before_{tb}_{ib}")
                model.Add(sa + da + scheduler.gap <= sb).OnlyEnforceIf(ba_before_b)
                model.Add(sb + db + scheduler.gap <= sa).OnlyEnforceIf(ba_before_b.Not())
                # This enforces one of the ordering; but if durations can be zero, both may hold; that's okay.

    # Blackout avoidance: for each task and blackout, enforce that run does not intersect blackout
    for key, tv in task_vars.items():
        s_var = tv["s"]; d_var = tv["d"]
        for bi, b in enumerate(scheduler.blackouts):
            b_before = model.NewBoolVar(f"{key}_before_blackout_{bi}")
            # run ends <= blackout.start OR run starts >= blackout.end
            model.Add(s_var + d_var <= b.start).OnlyEnforceIf(b_before)
            model.Add(s_var >= b.end).OnlyEnforceIf(b_before.Not())
            # We don't require either boolean to be true explicitly; but above OnlyEnforceIf requires the literal to control add.
            # To ensure one holds, add equality constraint
            model.AddBoolOr([b_before, b_before.Not()])  # tautology, left for readability

    # Objective: maximize sum(priority[type] * d_var)
    objective_terms = []
    for key, tv in task_vars.items():
        t = tv["type"]
        priority = priority_map.get(t,1)
        objective_terms.append(priority * tv["d"])
    model.Maximize(sum(objective_terms))

    # Solve
    solver = cp_model.CpSolver()
    if solver_time_limit_s is not None:
        solver.parameters.max_time_in_seconds = solver_time_limit_s
    solver.parameters.num_search_workers = 8
    result = solver.Solve(model)

    if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return []  # no feasible schedule found

    # Load job_name mapping from jobs_to_schedule.csv
    import pandas as pd
    try:
        df_sched = pd.read_csv('data/jobs_to_schedule.csv')
        # Build a mapping: job_type -> list of job_names (in order)
        job_name_map = {}
        for t in df_sched['job_type'].unique():
            job_name_map[t] = list(df_sched[df_sched['job_type'] == t]['job_name'])
        job_name_idx = {t: 0 for t in job_name_map}
    except Exception:
        job_name_map = {}
        job_name_idx = {}

    # Extract scheduled jobs: only those with duration > 0
    scheduled = []
    for key, tv in task_vars.items():
        s_val = solver.Value(tv["s"])
        d_val = solver.Value(tv["d"])
        t = tv["type"]
        if d_val > 0:
            job = Job(t, s_val, s_val + d_val, priority=priority_map.get(t,1), source="scheduled")
            # Assign display_name from job_name_map if available
            if t in job_name_map and job_name_idx[t] < len(job_name_map[t]):
                job.display_name = job_name_map[t][job_name_idx[t]]
                job_name_idx[t] += 1
            else:
                job.display_name = t
            scheduled.append(job)
    # Add scheduled to scheduler (optional: we don't mutate input scheduler here)
    return scheduled

# ---------------------------
# Unified entrypoint & metrics
# ---------------------------
def run_with_priority(scheduler: Scheduler,
                      first_end_map: Dict[str,int],
                      num_runs_per_type: Dict[str,int],
                      priority_map: Dict[str,int],
                      strategy: str = "ordered",   # "ordered", "weighted", "optimal"
                      per_run_max_cap: Optional[int]=None,
                      metrics: bool=True,
                      **strategy_kwargs) -> Dict[str, Any]:
    """
    Unified API. Returns dict containing:
      - scheduled_jobs: List[Job]
      - metrics: {...} (if metrics=True)
    """
    # Make a copy of scheduler to avoid mutating user's original: operate on a cloned scheduler
    sched_copy = Scheduler(list(scheduler.jobs), list(scheduler.blackouts), scheduler.gap, scheduler.horizon)
    if strategy == "ordered":
        result = schedule_priority_ordered(sched_copy, dict(first_end_map), dict(num_runs_per_type), dict(priority_map), per_run_max_cap=per_run_max_cap)
        scheduled = result['jobs']
        display_map = result['display_map']
    elif strategy == "weighted":
        scheduled = schedule_priority_weighted(sched_copy, dict(first_end_map), dict(num_runs_per_type), dict(priority_map), per_run_max_cap=per_run_max_cap, **strategy_kwargs)
        display_map = {}
    elif strategy == "optimal":
        scheduled = schedule_priority_optimal(sched_copy, dict(first_end_map), dict(num_runs_per_type), dict(priority_map), per_run_max_cap=per_run_max_cap, **strategy_kwargs)
        display_map = {}
    else:
        raise ValueError("Unknown strategy: choose 'ordered', 'weighted', or 'optimal'")
    # Optionally compute metrics
    metrics_out = {}
    if metrics:
        total = sum(j.end - j.start for j in scheduled)
        metrics_out['total_scheduled_minutes'] = total
        metrics_out['num_jobs'] = len(scheduled)
    return {"scheduled_jobs": scheduled, "metrics": metrics_out, "display_map": display_map}

    out = {"scheduled_jobs": scheduled, "display_map": display_map}
    if metrics:
        out["metrics"] = compute_metrics(scheduler, scheduled, priority_map, first_end_map)
    return out

def compute_metrics(scheduler: Scheduler, scheduled_jobs: List[Job], priority_map: Dict[str,int], first_end_map: Dict[str,int]) -> Dict[str,Any]:
    """
    Compute a set of metrics for analysis:
      - total_minutes: sum durations
      - weighted_score: sum(priority * duration)
      - per_type_minutes, per_type_score
      - avg_duration, num_runs
      - window utilization: (scheduled minutes) / (sum of free window minutes across types based on first_end_map)
    """
    mins = [j.end - j.start for j in scheduled_jobs]
    total_minutes = sum(mins)
    weighted_score = sum((priority_map.get(j.job_type,1) * (j.end - j.start)) for j in scheduled_jobs)
    per_type_minutes = {}
    per_type_score = {}
    for j in scheduled_jobs:
        per_type_minutes[j.job_type] = per_type_minutes.get(j.job_type, 0) + (j.end - j.start)
        per_type_score[j.job_type] = per_type_score.get(j.job_type, 0) + (priority_map.get(j.job_type,1)*(j.end - j.start))
    # compute available free minutes sum across types (based on feasible windows ignoring scheduled runs)
    total_free = 0
    for t in set([j.job_type for j in scheduler.jobs] + list(first_end_map.keys())):
        windows = scheduler.feasible_second_run_windows(t, first_end_map.get(t,0), include_scheduled=False)
        for win, maxdur in windows:
            total_free += win.duration()
    utilization = total_minutes / total_free if total_free > 0 else None
    avg_duration = statistics.mean(mins) if mins else 0
    num_runs = len(scheduled_jobs)
    metrics = {
        "total_minutes": total_minutes,
        "weighted_score": weighted_score,
        "per_type_minutes": per_type_minutes,
        "per_type_score": per_type_score,
        "avg_duration": avg_duration,
        "num_runs": num_runs,
        "total_free_minutes_approx": total_free,
        "utilization_ratio": utilization,
    }
    return metrics

# ---------------------------
# Export & visualization helpers
# ---------------------------
def export_schedule_to_csv(jobs: List[Job]) -> str:
    # Accept jobs as a list or a dict with display_map
    if isinstance(jobs, dict) and 'jobs' in jobs and 'display_map' in jobs:
        display_map = jobs['display_map']
        jobs = jobs['jobs']
    else:
        display_map = getattr(jobs, '_display_map', {})
    import pytz
    from datetime import datetime
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["job_name", "job_type", "start_min", "end_min", "start_time", "end_time", "duration_min", "priority", "source", "date", "timezone"])
    # Get current date and timezone
    tz_est = pytz.timezone('US/Eastern')
    tz_ist = pytz.timezone('Asia/Kolkata')
    base_date_est = datetime.now(tz_est).replace(hour=0, minute=0, second=0, microsecond=0)
    base_date_ist = datetime.now(tz_ist).replace(hour=0, minute=0, second=0, microsecond=0)
    for j in jobs:
        # Use display_name if present, else fallback
        job_name = getattr(j, 'display_name', display_map.get(j.job_type, j.job_type))
        # Output type A jobs in IST, others in EST
        if j.job_type == 'A':
            start_day = j.start // 1440
            job_date = base_date_ist + timedelta(days=start_day)
            date_str = job_date.strftime('%a %b %d, %Y')
            tz_str = job_date.tzname()
            start_time = fmt_time(j.start % 1440)
            end_time = fmt_time(j.end % 1440)
        else:
            start_day = j.start // 1440
            job_date = base_date_est + timedelta(days=start_day)
            date_str = job_date.strftime('%a %b %d, %Y')
            tz_str = job_date.tzname()
            start_time = fmt_time(j.start % 1440)
            end_time = fmt_time(j.end % 1440)
        writer.writerow([
            job_name, j.job_type, j.start, j.end, start_time, end_time,
            j.end - j.start, j.priority, j.source, date_str, tz_str
        ])
    return output.getvalue()

def export_schedule_to_json(jobs: List[Job]) -> str:
    # Accept jobs as a list or a dict with display_map
    if isinstance(jobs, dict) and 'jobs' in jobs and 'display_map' in jobs:
        display_map = jobs['display_map']
        jobs = jobs['jobs']
    else:
        display_map = getattr(jobs, '_display_map', {})
    arr = []
    for j in jobs:
        job_name = getattr(j, 'display_name', display_map.get(j.job_type, j.job_type))
        arr.append({
            "job_name": job_name,
            "job_type": j.job_type,
            "start_min": j.start,
            "end_min": j.end,
            "start_time": fmt_time(j.start),
            "end_time": fmt_time(j.end),
            "duration_min": j.end - j.start,
            "priority": j.priority,
            "source": j.source
        })
    return json.dumps(arr, indent=2)

import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta, datetime

def minutes_to_clock(mins: int) -> str:
    """Convert minutes after midnight to HH:MM format."""
    return f"{mins//60:02d}:{mins%60:02d}"

def get_base_type(job_type):
    # Always use only the leading capital letters as the base type
    m = re.match(r"^([A-Z]+)", job_type)
    return m.group(1) if m else job_type

def plot_gantt_with_job_labels(jobs, title="Schedule", blackout_intervals=None):
    print("[DEBUG] plot_gantt_with_job_labels CALLED")
    # Build rows with base_type and label
    rows = []
    for j in jobs:
        base_type = get_base_type(j.job_type)
        if hasattr(j, 'display_name'):
            label = j.display_name + ('*' if getattr(j, 'source', None) == 'existing' else '')
        else:
            if getattr(j, 'source', None) == 'existing':
                label = f"{j.job_type}*"
            else:
                label = j.job_type
        rows.append({
            "job_type": j.job_type,
            "base_type": base_type,
            "start": j.start,
            "duration": j.end - j.start,
            "label": label
        })

    if len(rows) == 0:
        print("No jobs to plot.")
        return
    df = pd.DataFrame(rows)

    # Debug output for axis/job info
    min_start = min(df['start'])
    max_end = max(df['start'] + df['duration'])
    min_day = min_start // 1440
    max_day = (max_end - 1) // 1440
    num_days = max_day - min_day + 1
    print("[DEBUG] Plot axis/job info:")
    print(f"  min_start: {min_start}")
    print(f"  max_end:   {max_end}")
    print(f"  min_day:   {min_day}")
    print(f"  max_day:   {max_day}")
    print(f"  num_days:  {num_days}")
    print(f"  num_jobs:  {len(df)}")
    # Build rows with base_type and label
    rows = []
    for j in jobs:
        base_type = get_base_type(j.job_type)
        if hasattr(j, 'display_name'):
            label = j.display_name + ('*' if getattr(j, 'source', None) == 'existing' else '')
        else:
            if getattr(j, 'source', None) == 'existing':
                label = f"{j.job_type}*"
            else:
                label = j.job_type
        rows.append({
            "job_type": j.job_type,
            "base_type": base_type,
            "start": j.start,
            "duration": j.end - j.start,
            "label": label
        })

    if len(rows) == 0:
        print("No jobs to plot.")
        return
    df = pd.DataFrame(rows)
    base_types = sorted(df["base_type"].unique())
    y_positions = {t: idx for idx, t in enumerate(base_types)}



    # Determine the full time span needed (in minutes)
    min_start = min(df['start'])
    max_end = max(df['start'] + df['duration'])
    # Round down min_start to nearest 60, up max_end to nearest 60
    min_start = (min_start // 60) * 60
    max_end = ((max_end + 59) // 60) * 60
    # Restore classic logic: width proportional to days, min 12, max 20
    min_day = min_start // 1440
    max_day = (max_end - 1) // 1440
    num_days = max_day - min_day + 1
    fig_width = min(20, max(12, num_days * 5))
    plt.figure(figsize=(fig_width, 2 + len(base_types) * 0.7))
    import matplotlib.colors as mcolors
    color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    color_map = {t: color_list[i % len(color_list)] for i, t in enumerate(base_types)}
    ax = plt.gca()

    # Set xlim to cover all scheduled days (midnight to midnight)
    ax.set_xlim(min_day * 1440, (max_day + 1) * 1440)

    if blackout_intervals is not None:
        for interval in blackout_intervals:
            ax.axvspan(interval.start, interval.end, ymin=0, ymax=1, color='grey', alpha=0.3, zorder=0)

    # Draw vertical lines for day boundaries at every midnight
    for d in range(min_day, max_day + 2):
        x = d * 1440
        ax.axvline(x=x, color='k', linestyle=':', alpha=0.3, zorder=0)

    # Plot bars with different colors per base_type, grouped by base_type
    for _, row in df.iterrows():
        y = y_positions[row["base_type"]]
        ax.barh(
            y=y,
            width=row["duration"],
            left=row["start"],
            height=0.6,
            align="center",
            color=color_map[row["base_type"]],
            zorder=1
        )
        ax.text(
            x=row["start"] + row["duration"] / 2,
            y=y,
            s=row["label"],
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
            zorder=2
        )

    # Set y-ticks to base_type (all jobs of same base_type share the same line)
    plt.yticks(list(y_positions.values()), list(y_positions.keys()))

    # Add a text box at the top left showing the scheduling algorithm in use (from title)
    plt.gca().text(
        0.01, 0.98, f"Algorithm: {title}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    # Y-axis
    plt.yticks(list(y_positions.values()), list(y_positions.keys()))



    # Add secondary x-axis for IST (EST + 10.5 hours)
    def est_to_ist(mins):
        mins_ist = (mins + 630) % 1440
        h = mins_ist // 60
        m = mins_ist % 60
        return f"{h:02d}:{m:02d}"

    # X-ticks: every hour across all days
    xticks = list(range(min_day * 1440, (max_day + 1) * 1440 + 1, 60))
    ax.set_xticks(xticks)
    ax.set_xticklabels([minutes_to_clock(x % 1440) for x in xticks], rotation=90)
    secax = ax.secondary_xaxis('top')
    secax.set_xticks(xticks)
    secax.set_xticklabels([est_to_ist(x % 1440) for x in xticks], rotation=90)
    secax.set_xlabel('Time of Day (IST)')
    plt.xlabel("Time of Day (EST)")
    plt.title(title)

    plt.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_facecolor("white")
    plt.tight_layout()
    plt.show()
    
# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":

    import os
    import pandas as pd

    # File paths (edit as needed)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    existing_jobs_csv = os.path.join(data_dir, 'existing_jobs.csv')
    jobs_to_schedule_csv = os.path.join(data_dir, 'jobs_to_schedule.csv')
    import datetime
    blackout_intervals_file = os.path.join(data_dir, 'blackout_intervals.txt')


    # 1. Load blackout intervals from crontab-like file FIRST
    import re
    import pytz
    timezone = 'EST'
    with open(blackout_intervals_file) as f:
        lines = f.readlines()
    for l in lines:
        if l.strip().upper().startswith('# TIMEZONE:'):
            timezone = l.strip().split(':', 1)[1].strip().upper()
            break
    try:
        tz = pytz.timezone(timezone)
    except Exception:
        print(f"[WARNING] Unknown timezone '{timezone}', defaulting to EST")
        tz = pytz.timezone('EST')
        timezone = 'EST'

    # Determine scheduling horizon (default 14 days)
    horizon_days = 14
    # Optionally, infer from jobs to schedule
    if os.path.exists(jobs_to_schedule_csv):
        df_sched_tmp = pd.read_csv(jobs_to_schedule_csv)
        max_job_end = 0
        for _, row in df_sched_tmp.iterrows():
            if 'duration' in row:
                max_job_end = max(max_job_end, int(row.get('duration', 0)))
        horizon_days = max(horizon_days, (max_job_end // 1440) + 7)

    blackout_minutes_all = set()
    blackouts = []
    import datetime
    for day_offset in range(horizon_days):
        # Compute the date for this day in the main timezone
        base_date = (datetime.date.today() + datetime.timedelta(days=day_offset))
        for line in lines:
            orig_line = line
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Parse crontab: minute hour day_of_month month day_of_week [TIMEZONE]
            parts = re.split(r'\s+', line)
            if len(parts) < 5:
                continue
            minute, hour, dom, month, dow = parts[:5]
            line_tz = timezone
            if len(parts) > 5:
                line_tz = parts[5].upper()
            try:
                tz_line = pytz.timezone(line_tz) if line_tz != 'IST' else pytz.timezone('Asia/Kolkata')
            except Exception:
                print(f"[WARNING] Unknown timezone '{line_tz}', defaulting to EST")
                tz_line = pytz.timezone('EST')
                line_tz = 'EST'
            # Compute the local date for this day in the blackout's timezone
            local_date = base_date
            # Compute crontab weekday for this date in the blackout's timezone
            dt_local_midnight = tz_line.localize(datetime.datetime.combine(local_date, datetime.time(0, 0)))
            py_dow = dt_local_midnight.weekday()
            if py_dow == 6:
                this_dow = 0  # Sunday
            else:
                this_dow = py_dow + 1
            this_dom = local_date.day
            this_month = local_date.month
            def expand(field, minval, maxval):
                if field == '*':
                    return list(range(minval, maxval+1))
                vals = []
                for part in field.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        vals.extend(range(start, end+1))
                    else:
                        vals.append(int(part))
                return vals
            minutes = expand(minute, 0, 59)
            hours = expand(hour, 0, 23)
            doms = expand(dom, 1, 31) if dom != '*' else [this_dom]
            months = expand(month, 1, 12) if month != '*' else [this_month]
            if dow == '*':
                days_of_week = [this_dow]
            else:
                days_of_week = expand(dow, 0, 6)
            # Only apply if this day matches all fields
            if (this_dom in doms) and (this_month in months) and (this_dow in days_of_week):
                print(f"[DEBUG] Applying blackout line: '{orig_line.strip()}' for {local_date} (crontab weekday={this_dow}, timezone={line_tz})")
                for h in hours:
                    for m in minutes:
                        # Convert this minute in line_tz to EST minutes since midnight, offset by day
                        dt_local = tz_line.localize(datetime.datetime.combine(local_date, datetime.time(h, m)))
                        dt_est = dt_local.astimezone(pytz.timezone('EST'))
                        est_min = (day_offset * 1440) + (dt_est.hour * 60 + dt_est.minute)
                        blackout_minutes_all.add(est_min)
    # Merge consecutive blackout minutes into intervals (for all days)
    blackout_minutes = sorted(blackout_minutes_all)
    if blackout_minutes:
        start = prev = blackout_minutes[0]
        for minute in blackout_minutes[1:]:
            if minute == prev + 1:
                prev = minute
            else:
                blackouts.append(Interval(start, prev + 1))
                start = prev = minute
        blackouts.append(Interval(start, prev + 1))
    print(f"[DEBUG] Blackout intervals for all days: {[ (iv.start, iv.end) for iv in blackouts ]}")

    # 1b. Load existing jobs (with job_name column)
    import pytz
    ist_tz = pytz.timezone('Asia/Kolkata')
    jobs_tz = 'IST'
    with open(existing_jobs_csv) as f:
        lines = f.readlines()
    # Check for TIMEZONE header
    if lines[0].strip().upper().startswith('A,TIMEZONE:'):
        jobs_tz = lines[0].strip().split(':', 1)[1].strip().upper()
        csv_start = 1
    else:
        csv_start = 0
    df_exist = pd.read_csv(existing_jobs_csv, skiprows=csv_start)
    try:
        tz = pytz.timezone(jobs_tz) if jobs_tz != 'IST' else ist_tz
    except Exception:
        print(f"[WARNING] Unknown timezone '{jobs_tz}', defaulting to IST")
        tz = ist_tz
        jobs_tz = 'IST'
    existing_jobs = []
    import pytz, datetime
    today = datetime.date.today()
    for _, row in df_exist.iterrows():
        # Parse start as hh:mm and timezone per job
        start_str = str(row['start'])
        if ':' in start_str:
            h, m = map(int, start_str.split(':'))
        else:
            h, m = divmod(int(start_str), 60)
        dur_str = str(row['duration'])
        if ':' in dur_str:
            dh, dm = map(int, dur_str.split(':'))
            duration_local = dh * 60 + dm
        else:
            duration_local = int(dur_str)
        job_tz = str(row.get('timezone', 'EST')).upper()
        try:
            if job_tz == 'IST':
                tz_job = pytz.timezone('Asia/Kolkata')
            else:
                tz_job = pytz.timezone(job_tz)
        except Exception:
            tz_job = pytz.timezone('EST')
            job_tz = 'EST'
        # Create a datetime in the job's timezone for today at hh:mm
        dt_start_local = tz_job.localize(datetime.datetime(today.year, today.month, today.day, h, m, 0, 0))
        dt_end_local = dt_start_local + datetime.timedelta(minutes=duration_local)
        est_tz = pytz.timezone('EST')
        dt_start_est = dt_start_local.astimezone(est_tz)
        dt_end_est = dt_end_local.astimezone(est_tz)
        start_est = dt_start_est.hour * 60 + dt_start_est.minute
        end_est = dt_end_est.hour * 60 + dt_end_est.minute
        job = Job(
            row['job_type'],
            start_est,
            end_est,
            int(row.get('priority', 1)),
            'existing'
        )
        job.group_type = row.get('type', '')
        job.display_name = row['job_name'] if 'job_name' in row else row['job_type']
        # Check blackout overlap (in EST minutes)
        overlaps_blackout = False
        for b in blackouts:
            if not (end_est <= b.start or start_est >= b.end):
                overlaps_blackout = True
                break
        if overlaps_blackout:
            print(f"[WARNING] Existing job {job.display_name} ({job.job_type}) overlaps blackout and will be skipped.")
            continue
        existing_jobs.append(job)

    # 2. Load jobs to schedule (each run as a row: job_type, type, priority)
    df_sched = pd.read_csv(jobs_to_schedule_csv)
    df_sched['base_job_type'] = df_sched['job_type'].str.extract(r'^([A-Za-z]+)')
    # Count number of runs per base_job_type
    from collections import Counter, defaultdict
    num_runs_counter = Counter(df_sched['base_job_type'])
    num_runs = dict(num_runs_counter)
    # For priority, use the most common priority per base_job_type (or the first if mixed)
    priority_map = {}
    for job_type in num_runs:
        priorities = df_sched[df_sched['base_job_type'] == job_type]['priority']
        priority_map[job_type] = int(priorities.mode().iloc[0])
    # Store group_type for each job_type for display
    group_type_map = {}
    for _, row in df_sched.iterrows():
        group_type_map[row['job_type']] = row.get('type', '')

    # 3. Load blackout intervals from crontab-like file

    import re
    import pytz
    timezone = 'EST'
    with open(blackout_intervals_file) as f:
        lines = f.readlines()
    for l in lines:
        if l.strip().upper().startswith('# TIMEZONE:'):
            timezone = l.strip().split(':', 1)[1].strip().upper()
            break
    try:
        tz = pytz.timezone(timezone)
    except Exception:
        print(f"[WARNING] Unknown timezone '{timezone}', defaulting to EST")
        tz = pytz.timezone('EST')
        timezone = 'EST'

    # Determine scheduling horizon (default 14 days)
    horizon_days = 14
    # Optionally, infer from jobs to schedule
    if not df_sched.empty:
        max_job_end = 0
        for _, row in df_sched.iterrows():
            if 'duration' in row:
                max_job_end = max(max_job_end, int(row.get('duration', 0)))
        horizon_days = max(horizon_days, (max_job_end // 1440) + 7)

    blackout_minutes_all = set()
    blackouts = []
    for day_offset in range(horizon_days):
        # Compute the date for this day in the main timezone
        base_date = (datetime.date.today() + datetime.timedelta(days=day_offset))
        for line in lines:
            orig_line = line
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Parse crontab: minute hour day_of_month month day_of_week [TIMEZONE]
            parts = re.split(r'\s+', line)
            if len(parts) < 5:
                continue
            minute, hour, dom, month, dow = parts[:5]
            line_tz = timezone
            if len(parts) > 5:
                line_tz = parts[5].upper()
            try:
                tz_line = pytz.timezone(line_tz) if line_tz != 'IST' else pytz.timezone('Asia/Kolkata')
            except Exception:
                print(f"[WARNING] Unknown timezone '{line_tz}', defaulting to EST")
                tz_line = pytz.timezone('EST')
                line_tz = 'EST'
            # Compute the local date for this day in the blackout's timezone
            local_date = base_date
            # Compute crontab weekday for this date in the blackout's timezone
            dt_local_midnight = tz_line.localize(datetime.datetime.combine(local_date, datetime.time(0, 0)))
            py_dow = dt_local_midnight.weekday()
            if py_dow == 6:
                this_dow = 0  # Sunday
            else:
                this_dow = py_dow + 1
            this_dom = local_date.day
            this_month = local_date.month
            def expand(field, minval, maxval):
                if field == '*':
                    return list(range(minval, maxval+1))
                vals = []
                for part in field.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        vals.extend(range(start, end+1))
                    else:
                        vals.append(int(part))
                return vals
            minutes = expand(minute, 0, 59)
            hours = expand(hour, 0, 23)
            doms = expand(dom, 1, 31) if dom != '*' else [this_dom]
            months = expand(month, 1, 12) if month != '*' else [this_month]
            if dow == '*':
                days_of_week = [this_dow]
            else:
                days_of_week = expand(dow, 0, 6)
            # Only apply if this day matches all fields
            if (this_dom in doms) and (this_month in months) and (this_dow in days_of_week):
                print(f"[DEBUG] Applying blackout line: '{orig_line.strip()}' for {local_date} (crontab weekday={this_dow}, timezone={line_tz})")
                for h in hours:
                    for m in minutes:
                        # Convert this minute in line_tz to EST minutes since midnight, offset by day
                        dt_local = tz_line.localize(datetime.datetime.combine(local_date, datetime.time(h, m)))
                        dt_est = dt_local.astimezone(pytz.timezone('EST'))
                        est_min = (day_offset * 1440) + (dt_est.hour * 60 + dt_est.minute)
                        blackout_minutes_all.add(est_min)
    # Merge consecutive blackout minutes into intervals (for all days)
    blackout_minutes = sorted(blackout_minutes_all)
    if blackout_minutes:
        start = prev = blackout_minutes[0]
        for minute in blackout_minutes[1:]:
            if minute == prev + 1:
                prev = minute
            else:
                blackouts.append(Interval(start, prev + 1))
                start = prev = minute
        blackouts.append(Interval(start, prev + 1))
    print(f"[DEBUG] Blackout intervals for all days: {[ (iv.start, iv.end) for iv in blackouts ]}")

    # 4. Compute horizon: cover all blackout intervals (all days)
    if blackouts:
        max_blackout_end = max(b.end for b in blackouts)
        horizon = Interval(0, max(24*60, max_blackout_end))
    else:
        horizon = Interval(0, 24*60)
    gap = 15
    # 5. Compute first_end_map from existing jobs (latest end per type), default to 0 for new types
    first_end_map = {}
    for job in existing_jobs:
        if job.job_type not in first_end_map or job.end > first_end_map[job.job_type]:
            first_end_map[job.job_type] = job.end
    for job_type in num_runs:
        if job_type not in first_end_map:
            first_end_map[job_type] = 0
    scheduler = Scheduler(existing_jobs.copy(), blackouts, gap, horizon)

    out_ord = run_with_priority(scheduler, first_end_map, num_runs, priority_map, strategy="ordered", per_run_max_cap=240)
    def window_value_fn(window, maxdur, t):
        return maxdur + (1440 - window.start) / 1000.0
    out_w = run_with_priority(scheduler, first_end_map, num_runs, priority_map, strategy="weighted", per_run_max_cap=180, metrics=True, window_value_fn=window_value_fn)
    out_opt = None
    while True:
        print("\nMenu:")
        print("1. Show scheduled jobs for all algorithms")
        print("2. Plot Gantt for Ordered schedule")
        print("3. Export Ordered scheduled jobs to CSV")
        print("4. Show number of days required to run all jobs (day-based plot)")
        print("0. Exit")
        choice = input("Select an option: ").strip()
        if choice == "1":
            import pandas as pd
            import math
            # Prepare data for all algorithms
            jobs_data = []
            algos = []
            def safe_display_name(job):
                dn = getattr(job, 'display_name', None)
                if dn is None or (isinstance(dn, float) and math.isnan(dn)):
                    return job.job_type
                return dn
            # Ordered
            for job in out_ord["scheduled_jobs"]:
                job_dict = {**job.__dict__, "algorithm": "Ordered"}
                job_dict["display_name"] = safe_display_name(job)
                jobs_data.append(job_dict)
            algos.append("Ordered")
            # Weighted
            for job in out_w["scheduled_jobs"]:
                job_dict = {**job.__dict__, "algorithm": "Weighted"}
                job_dict["display_name"] = safe_display_name(job)
                jobs_data.append(job_dict)
            algos.append("Weighted")
            # Optimal
            optimal_algo = "CP-SAT" if HAS_ORTOOLS else "(not available)"
            if out_opt is None and HAS_ORTOOLS:
                out_opt = run_with_priority(scheduler, first_end_map, num_runs, priority_map, strategy="optimal", per_run_max_cap=240)
            if out_opt and "scheduled_jobs" in out_opt and out_opt["scheduled_jobs"]:
                for job in out_opt["scheduled_jobs"]:
                    job_dict = {**job.__dict__, "algorithm": "Optimal (CP-SAT)"}
                    job_dict["display_name"] = safe_display_name(job)
                    jobs_data.append(job_dict)
                algos.append("Optimal (CP-SAT)")
            # Display info
            print("\nScheduled jobs for all algorithms:")
            if jobs_data:
                df_jobs = pd.DataFrame(jobs_data)
                # Reorder columns for clarity
                columns = [c for c in ["display_name", "job_type", "start", "end", "priority", "source", "algorithm"] if c in df_jobs.columns]
                if "start" in df_jobs.columns and "end" in df_jobs.columns:
                    df_jobs["start_time"] = df_jobs["start"].apply(lambda x: minutes_to_clock(x % 1440))
                    df_jobs["end_time"] = df_jobs["end"].apply(lambda x: minutes_to_clock(x % 1440))
                    columns += ["start_time", "end_time"]
                print(df_jobs[columns].to_string(index=False))
            else:
                print("No jobs scheduled by any algorithm.")
            print("\nAlgorithm(s) used:", ", ".join(algos))
            print(f"Optimal scheduling available: {'Yes' if HAS_ORTOOLS else 'No'} (Algorithm: {optimal_algo})")
            if out_opt and "metrics" in out_opt:
                print("Optimal schedule metrics:", out_opt["metrics"])
        elif choice == "2":
            print("\nPlotting ordered schedule Gantt (existing jobs marked with *)")
            all_jobs = existing_jobs + out_ord["scheduled_jobs"]
            if not all_jobs:
                print("No jobs to plot.")
            else:
                plot_gantt_with_job_labels(all_jobs, title="Ordered Strategy Gantt", blackout_intervals=blackouts)
        elif choice == "3":
            import os
            # Ensure output directory exists
            output_dir = os.path.join(os.path.dirname(__file__), "output")
            os.makedirs(output_dir, exist_ok=True)
            export_path = os.path.join(output_dir, "scheduled_jobs_output.csv")
            with open(export_path, "w") as f:
                f.write(export_schedule_to_csv(out_ord["scheduled_jobs"]))
            print(f"Exported to {export_path}")
        elif choice == "4":
            # Compute number of days required and show jobs grouped by BASE type (one row per type) in a Gantt chart
            print("[DEBUG] Entered menu option 4 handler (day-based plot)")
            print("[DEBUG] Entered menu option 4 handler (day-based plot)", file=sys.stderr)
            jobs = out_ord["scheduled_jobs"]
            if not jobs:
                print("No jobs to analyze.")
            else:
                import matplotlib.pyplot as plt
                import matplotlib.colors as mcolors
                import datetime
                import pytz
                import re
                from collections import defaultdict
                today = datetime.date.today()
                est_tz = pytz.timezone('US/Eastern')
                ist_tz = pytz.timezone('Asia/Kolkata')
                # Helper to extract base type (leading capital letters)
                def get_base_type(job_type):
                    m = re.match(r"^([A-Z]+)", job_type)
                    return m.group(1) if m else job_type

                # Group jobs by base type
                jobs_by_base = defaultdict(list)
                for j in jobs:
                    base = get_base_type(j.job_type)
                    jobs_by_base[base].append(j)
                # Sort base types for y-axis
                base_types = sorted(jobs_by_base.keys())
                y_labels = base_types
                y_label_map = {base: i for i, base in enumerate(base_types)}
                # Color map for jobs (by job instance)
                color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
                # X-axis: all days present
                all_days = set()
                plot_segments = []  # (base, job, day, seg_start, seg_end)
                for base in base_types:
                    for j in jobs_by_base[base]:
                        start_day = j.start // 1440
                        end_day = (j.end - 1) // 1440  # inclusive
                        for day in range(start_day, end_day + 1):
                            day_start = day * 1440
                            day_end = (day + 1) * 1440
                            seg_start = max(j.start, day_start)
                            seg_end = min(j.end, day_end)
                            all_days.add(day)
                            plot_segments.append((base, j, day, seg_start, seg_end))
                days = sorted(all_days)
                # Compute EST and IST date labels for each day
                day_labels = []
                for d in days:
                    # Base date in EST
                    base_dt_est = est_tz.localize(datetime.datetime.combine(today, datetime.time(0, 0))) + datetime.timedelta(days=d)
                    base_dt_ist = base_dt_est.astimezone(ist_tz)
                    label = f"{base_dt_est.strftime('%Y-%m-%d EST')}\n{base_dt_ist.strftime('%Y-%m-%d IST')}"
                    day_labels.append(label)

                # Collect all timezones used in jobs (for vertical lines)
                timezones_used = set(['US/Eastern', 'Asia/Kolkata'])
                for j in jobs:
                    if hasattr(j, 'timezone') and j.timezone:
                        timezones_used.add(j.timezone)
                # Map common names
                tz_map = {'EST': 'US/Eastern', 'IST': 'Asia/Kolkata'}
                for tz in list(timezones_used):
                    if tz in tz_map:
                        timezones_used.add(tz_map[tz])

                plt.figure(figsize=(max(8, len(days) * 2), 2 + len(y_labels) * 0.7))
                for idx, (base, j, day, seg_start, seg_end) in enumerate(plot_segments):
                    y = y_label_map[base]
                    left = day + (seg_start - day * 1440) / 1440.0
                    width = (seg_end - seg_start) / 1440.0
                    color = color_list[idx % len(color_list)]
                    plt.barh(y, width, left=left, height=0.6, color=color, zorder=1)
                    # Label with job display_name if present, else job_type
                    job_label = getattr(j, 'display_name', j.job_type)
                    # Add white outline for better visibility
                    text = plt.text(left + width/2, y, job_label, ha='center', va='center', color='black', fontsize=10, fontweight='bold', zorder=2)
                    try:
                        from matplotlib import patheffects
                        text.set_path_effects([
                            patheffects.Stroke(linewidth=2, foreground='white'),
                            patheffects.Normal()
                        ])
                    except ImportError:
                        pass

                # Draw vertical dotted lines for day transitions in each timezone
                for tz_name in sorted(timezones_used):
                    try:
                        import pytz
                        tz = pytz.timezone(tz_name)
                    except Exception:
                        continue
                    # For each day, compute the fractional x where the day boundary in this tz falls in EST day units
                    for d in days:
                        base_dt_est = est_tz.localize(datetime.datetime.combine(today, datetime.time(0, 0))) + datetime.timedelta(days=d)
                        # The next day in this timezone
                        next_day_tz = tz.localize(datetime.datetime.combine((today + datetime.timedelta(days=d)), datetime.time(0, 0))) + datetime.timedelta(days=1)
                        # Convert next day midnight in this tz to EST
                        next_day_midnight_est = next_day_tz.astimezone(est_tz)
                        # Compute the x position (in EST day units)
                        x = (next_day_midnight_est - base_dt_est).total_seconds() / (24*3600)
                        x_pos = d + x
                        plt.axvline(x=x_pos, color='k', linestyle=':', alpha=0.5, label=f'{tz_name} day')
                # Remove duplicate legend entries
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                # Place legend below the plot, right-aligned
                plt.legend(
                    by_label.values(),
                    by_label.keys(),
                    loc='lower center',
                    bbox_to_anchor=(1.0, -0.25),
                    fontsize=8,
                    ncol=1,  # Stack legend entries vertically
                    frameon=False,
                    borderaxespad=0.
                )

                plt.yticks(range(len(y_labels)), y_labels)
                plt.xticks(days, day_labels, rotation=30)
                plt.xlabel("Day (EST/IST)")
                plt.title("Jobs scheduled per day (grouped by base type)")
                plt.tight_layout()
                print("[DEBUG] About to call plt.show() for day-based plot", file=sys.stderr)
                print("[DEBUG] About to call plt.show() for day-based plot")
                plt.show()
        elif choice == "0":
            break
        else:
            print("Invalid option. Try again.")