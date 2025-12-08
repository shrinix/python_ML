# Example CSV headers for schedule_jobs.py
# Place these files in the same directory as schedule_jobs.py or specify the path in the script.

# 1. existing_jobs.csv
# job_type,start,end,priority
A,540,585,10
B,510,540,5
C,660,680,1

# 2. jobs_to_schedule.csv
# job_type,num_runs,priority
A,2,10
B,1,5
C,2,1

# 3. blackout_intervals.csv
# start,end
720,780
1110,1140
