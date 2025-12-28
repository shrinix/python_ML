# Job Scheduling System

This project implements a flexible job scheduling system with blackout enforcement, multiple scheduling algorithms, and visualization tools. It is designed for research, demo, and practical scheduling scenarios.

---

## Folder Structure

- `schedule_jobs.py` — Main script: scheduling logic, CLI, plotting, and data models.
- `data/`
  - `jobs_to_schedule.csv` — Jobs to be scheduled (job_type, job_name, priority, etc.)
  - `existing_jobs.csv` — Already scheduled jobs
  - `blackout_intervals.txt` — Crontab-like blackout intervals (no jobs can be scheduled)
- `output/`
  - `scheduled_jobs_output.csv` — Output of scheduled jobs
- `README_CSV_FORMATS.txt` — Example formats for the CSV files
- `debug_scheduling.md`, `scheduling_failure_report.md` — Debugging and failure logs

---

## How to Run

1. **Prepare your environment:**
   - Install Python 3.8+ and required packages:
     ```sh
     pip install pandas matplotlib ortools
     ```
   - Place your job and blackout data in the `data/` folder (see `README_CSV_FORMATS.txt` for formats).

2. **Run the scheduler:**
   ```sh
   python schedule_jobs.py
   ```

3. **Menu Options:**
   When you run the script, you will see a menu:

   - `1. Show scheduled jobs for all algorithms`
     - Prints a table of all jobs scheduled by each algorithm (Ordered, Weighted, Optimal if available).
   - `2. Plot Gantt for Ordered schedule`
     - Shows a Gantt chart of the ordered schedule (existing jobs marked with *).
   - `3. Export Ordered scheduled jobs to CSV`
     - Saves the ordered schedule to `output/scheduled_jobs_output.csv`.
   - `4. Show number of days required to run all jobs (day-based plot)`
     - Plots jobs grouped by base type, showing how many days are needed.
   - `0. Exit`
     - Exits the program.

---

## Code Structure & Logic

### Data Models
- **Interval**: Represents a time interval (start, end in minutes).
- **Job**: Represents a job (type, start, end, priority, source, display_name).

### Scheduler Class
- Manages jobs, blackout intervals, and scheduling constraints (gaps, horizon).
- Key methods:
  - `build_blocked_for_type`: Computes blocked intervals for a job type.
  - `feasible_second_run_windows`: Finds available windows for scheduling a job.
  - `add_scheduled_job`, `remove_scheduled_jobs`: Manage scheduled jobs.

### Scheduling Algorithms
- **schedule_priority_ordered**: Schedules jobs by type and priority, one type at a time.
- **schedule_priority_weighted**: Greedy, cross-type scheduling by weighted score.
- **schedule_priority_optimal**: Uses Google OR-Tools CP-SAT solver for optimal scheduling (if available).

All algorithms assign a `display_name` to each scheduled job, using the job's name from the CSV.

### Unified Entrypoint
- **run_with_priority**: Unified API to run any scheduling strategy and collect metrics.

### Export & Visualization
- **export_schedule_to_csv/json**: Export scheduled jobs to CSV/JSON.
- **plot_gantt_with_job_labels**: Gantt chart visualization of scheduled jobs, with blackout intervals and job labels.

### CLI & Menu
- Menu-driven CLI for:
  - Viewing scheduled jobs.
  - Plotting Gantt charts.
  - Exporting results.
  - Debugging and analysis.

---

## Data Flow

1. **Input**: Reads jobs and blackout intervals from CSV and text files in `data/`.
2. **Scheduling**: Applies the selected algorithm, respecting blackout intervals and job constraints.
3. **Output**: Displays results in the terminal, plots Gantt charts, and exports to CSV.

---

## Extensibility
- New scheduling strategies can be added by implementing a new function and wiring it into the menu and `run_with_priority`.
- Data files can be extended with more fields as needed.

---

## Debugging & Reporting
- Extensive debug output and error reporting (to stdout and stderr).
- Failure and debug logs are kept in markdown files for traceability.

---

## Example CSV Formats
See `README_CSV_FORMATS.txt` for sample input files.

---

## Notes
- The optimal scheduling algorithm requires OR-Tools (`pip install ortools`).
- All time calculations are in minutes since midnight EST unless otherwise noted.
- Type A jobs are displayed in IST in exports/plots; others in EST.

---

For further details, see comments in `schedule_jobs.py` or ask for a deep dive into any function or module.
