# Debugging Scheduling Issue: Type A Jobs Not Scheduled on Day 1

## Problem
Despite round-robin and per-job horizon extension, jobs of type A are not scheduled on day 1.

## Next Steps
- Output the actual scheduled jobs and their start times for all types.
- Print debug info for each job scheduling attempt: job name, type, attempted day, and why it was/wasn't scheduled.
- Check if blackout intervals, gaps, or job durations are blocking type A jobs on day 1.

## Action
Patch the scheduling code to print debug info for each job scheduling attempt.
