# Scheduling Failure Report for Type C Jobs

The following jobs of type C were not scheduled:

- C1
- C2
- C3
- C4

## Reason Analysis

For each job, the scheduler attempts to find a feasible time window after all previously scheduled jobs of the same type, considering:
- Blackout intervals (no-run times)
- Minimum gap between jobs
- 24-hour day limit
- Job durations and priorities

### Detailed Attempt Log
