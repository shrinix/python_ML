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
def schedule_priority_ordered(scheduler: Scheduler,
                              first_end_map: Dict[str,int],
                              num_runs_per_type: Dict[str,int],
                              priority_map: Dict[str,int],
                              per_run_max_cap: Optional[int]=None) -> List[Job]:
    # Helper to extract base_type (leading capital letters)
    def get_base_type(job_type):
        m = re.match(r"^([A-Z]+)", job_type)
        return m.group(1) if m else job_type
    """
    Simple ordered strategy:
      - sorts types by priority descending
      - for each type, schedules its runs greedily (earliest-first), updating timeline as we go
    """
    scheduled = []
    # sort by priority desc, tie-breaker by type name
    ordered_types = sorted(priority_map.keys(), key=lambda t: (-priority_map[t], t))
    fe = dict(first_end_map)
    # Build a set of existing job names for each type
    from collections import defaultdict
    existing_names = defaultdict(set)
    for j in scheduler.jobs:
        base = re.match(r"^([A-Z]+)", j.job_type).group(1) if re.match(r"^([A-Z]+)", j.job_type) else j.job_type
        existing_names[base].add(j.job_type)
    # Helper to generate next available name for scheduled jobs (A2, A3, ... skipping existing)
    def next_job_name(base, used, counter):
        while True:
            name = f"{base}{counter[0]}"
            counter[0] += 1
            if name not in used:
                used.add(name)
                return name

    scheduled_names = defaultdict(set)
    # Read the original job names for scheduled jobs from jobs_to_schedule.csv
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
    # For each type, get the list of job_names in order from jobs_to_schedule.csv
    sched_names_map = {}
    for t in ordered_types:
        sched_names_map[t] = list(df_sched[df_sched['job_type'] == t]['job_name'])

    # Maintain a mapping from internal job index to display name for scheduled jobs
    scheduled_job_display_map = {}

    for t in ordered_types:
        n = num_runs_per_type.get(t, 0)
        base = re.match(r"^([A-Z]+)", t).group(1) if re.match(r"^([A-Z]+)", t) else t
        used = set(existing_names[base]) | set(scheduled_names[base])
        sched_names = sched_names_map.get(t, [])
        for i in range(n):
            windows = scheduler.feasible_second_run_windows(t, fe[t], include_scheduled=True)
            if not windows:
                break
            win, maxdur = windows[0]  # earliest
            dur = maxdur if per_run_max_cap is None else min(maxdur, per_run_max_cap)
            if dur <= 0:
                break
            start = win.start
            end = start + dur
            # Enforce 24-hour constraint
            if start < 0 or end > 1440:
                break
            # Use an internal indexed name for scheduling, but map to display name
            internal_name = f"{base}_sched_{i+1}"
            display_name = sched_names[i] if i < len(sched_names) else f"{base}_extra_{i+1}"
            scheduled_job_display_map[internal_name] = display_name
            scheduled_names[base].add(internal_name)
            # Use t as job_type (from jobs_to_schedule.csv)
            job = Job(t, start, end, priority=priority_map.get(t,1), source="scheduled")
            job.display_name = display_name
            scheduled.append(job)
            scheduler.add_scheduled_job(job)
            fe[t] = job.end
    # Return both the jobs and the display map
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

    # Extract scheduled jobs: only those with duration > 0
    scheduled = []
    for key, tv in task_vars.items():
        s_val = solver.Value(tv["s"])
        d_val = solver.Value(tv["d"])
        t = tv["type"]
        if d_val > 0:
            job = Job(t, s_val, s_val + d_val, priority=priority_map.get(t,1), source="scheduled")
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
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["job_name", "job_type", "start_min", "end_min", "start_time", "end_time", "duration_min", "priority", "source"])
    for j in jobs:
        # Use display_name if present, else fallback
        job_name = getattr(j, 'display_name', display_map.get(j.job_type, j.job_type))
        writer.writerow([job_name, j.job_type, j.start, j.end, fmt_time(j.start), fmt_time(j.end), j.end - j.start, j.priority, j.source])
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

def plot_gantt_with_job_labels(jobs, title="Schedule"):
    """
    Gantt plot where:
      - X-axis shows actual clock times (HH:MM)
      - Bars are white-background, standard Matplotlib styling
      - Job labels (A1, A2...) printed inside bars in bold white
      - Y-axis displays job types (A, B, C...)
    """
    if not jobs:
        print("No jobs to plot.")
        return

    # Build table for plotting
    import re
    rows = []

    # Sort jobs by base_type and start time, but preserve full job_type for labels
    # Accept jobs as a list or a dict with display_map
    if isinstance(jobs, dict) and 'jobs' in jobs and 'display_map' in jobs:
        display_map = jobs['display_map']
        jobs = jobs['jobs']
    else:
        display_map = getattr(jobs, '_display_map', {})
    for j in sorted(jobs, key=lambda x: (get_base_type(x.job_type), x.start, x.job_type)):
        base_type = get_base_type(j.job_type)
        if hasattr(j, 'display_name'):
            label = j.display_name + ('*' if getattr(j, 'source', None) == 'existing' else '')
        else:
            if getattr(j, 'source', None) == 'existing':
                label = f"{j.job_type}*"
            else:
                label = display_map.get(j.job_type, j.job_type)
        rows.append({
            "job_type": j.job_type,
            "base_type": base_type,
            "start": j.start,
            "duration": j.end - j.start,
            "label": label
        })

    # Group jobs by base_type for y-axis (all jobs of same base_type share the same y position)
    base_types = sorted(set(row["base_type"] for row in rows))
    y_positions = {t: idx for idx, t in enumerate(base_types)}

    # Debug printout for grouping verification
    print("\n[DEBUG] Gantt Plot Grouping:")
    for row in rows:
        print(f"  job_type: {row['job_type']}, base_type: {row['base_type']}, label: {row['label']}")
    print(f"  y_positions (base_type to y): {y_positions}")
    if not jobs:
        print("No jobs to plot.")
        return

    df = pd.DataFrame(rows)

    # Group jobs by base_type for y-axis (all jobs of same base_type share the same y position)
    base_types = sorted(df["base_type"].unique())
    y_positions = {t: idx for idx, t in enumerate(base_types)}

    plt.figure(figsize=(14, 2 + len(base_types) * 0.7))

    # Assign a unique color to each base_type
    import matplotlib.colors as mcolors
    color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    color_map = {t: color_list[i % len(color_list)] for i, t in enumerate(base_types)}

    ax = plt.gca()

    # Plot blackout intervals as grey background bars
def plot_gantt_with_job_labels(jobs, title="Schedule", blackout_intervals=None):
    import pandas as pd
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

    plt.figure(figsize=(14, 2 + len(base_types) * 0.7))

    # Assign a unique color to each base_type
    import matplotlib.colors as mcolors
    color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    color_map = {t: color_list[i % len(color_list)] for i, t in enumerate(base_types)}

    ax = plt.gca()
    if blackout_intervals is not None:
        for interval in blackout_intervals:
            ax.axvspan(interval.start, interval.end, ymin=0, ymax=1, color='grey', alpha=0.3, zorder=0)

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


    # Show the entire 24-hour day on the x-axis
    ax = plt.gca()
    ax.set_xlim(0, 1440)
    xticks = list(range(0, 1441, 60))
    ax.set_xticks(xticks)
    ax.set_xticklabels([minutes_to_clock(x) for x in xticks])
    plt.xlabel("Time of Day (EST)")
    plt.title(title)

    # Add secondary x-axis for IST (EST + 10.5 hours)
    def est_to_ist(mins):
        mins_ist = (mins + 630) % 1440
        h = mins_ist // 60
        m = mins_ist % 60
        return f"{h:02d}:{m:02d}"

    secax = ax.secondary_xaxis('top')
    secax.set_xticks(xticks)
    secax.set_xticklabels([est_to_ist(x) for x in xticks])
    secax.set_xlabel('Time of Day (IST)')

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
    blackout_intervals_csv = os.path.join(data_dir, 'blackout_intervals.csv')

    # 1. Load existing jobs (with job_name column)
    df_exist = pd.read_csv(existing_jobs_csv)
    existing_jobs = []
    for _, row in df_exist.iterrows():
        job = Job(
            row['job_type'],
            int(row['start']),
            int(row['end']),
            int(row.get('priority', 1)),
            'existing'
        )
        job.group_type = row.get('type', '')
        # Attach display_name from job_name column
        job.display_name = row['job_name'] if 'job_name' in row else row['job_type']
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

    # 3. Load blackout intervals
    df_black = pd.read_csv(blackout_intervals_csv)
    blackouts = [Interval(int(row['start']), int(row['end'])) for _, row in df_black.iterrows()]

    # 4. Compute horizon (default: 0 to 1440 min)
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
    if HAS_ORTOOLS:
        out_opt = run_with_priority(scheduler, first_end_map, num_runs, priority_map, strategy="optimal", per_run_max_cap=240, metrics=True, solver_time_limit_s=10)
    else:
        out_opt = None

    # Menu
    while True:
        print("\nMenu:")
        print("1. Show Ordered scheduled jobs")
        print("2. Show Weighted scheduled jobs")
        print("3. Show Optimal scheduled jobs" + (" (available)" if HAS_ORTOOLS else " (not available)"))
        print("4. Plot Gantt for Ordered schedule")
        print("5. Export Ordered scheduled jobs to CSV")
        print("0. Exit")
        choice = input("Select an option: ").strip()
        if choice == "1":
            print("\nOrdered scheduled jobs:")
            print(export_schedule_to_csv(out_ord["scheduled_jobs"]))
            print("Metrics:", out_ord["metrics"])
        elif choice == "2":
            print("\nWeighted scheduled jobs:")
            print(export_schedule_to_csv(out_w["scheduled_jobs"]))
            print("Metrics:", out_w["metrics"])
        elif choice == "3":
            if out_opt:
                print("\nOptimal scheduled jobs:")
                print(export_schedule_to_csv(out_opt["scheduled_jobs"]))
                print("Metrics:", out_opt["metrics"])
            else:
                print("Optimal strategy not available (install ortools to enable).")
        elif choice == "4":
            print("\nPlotting ordered schedule Gantt (existing jobs marked with *)")
            all_jobs = existing_jobs + out_ord["scheduled_jobs"]
            if not all_jobs:
                print("No jobs to plot.")
            else:
                plot_gantt_with_job_labels(all_jobs, title="Ordered Strategy Gantt", blackout_intervals=blackouts)
        elif choice == "5":
            export_path = input("Enter filename to export (default: scheduled_jobs_output.csv): ").strip()
            if not export_path:
                export_path = "scheduled_jobs_output.csv"
            with open(export_path, "w") as f:
                f.write(export_schedule_to_csv(out_ord["scheduled_jobs"]))
            print(f"Exported to {export_path}")
        elif choice == "0":
            print("Exiting.")
            break
        else:
            print("Invalid option. Try again.")