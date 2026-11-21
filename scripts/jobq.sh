#!/bin/bash
# Serial job queue: ONE GPU job at a time (the laptop has 16 GB shared memory; parallel jobs crashed it).
#   append a job:  echo 'uv run python ... > artifacts/runs/logs/x.txt 2>&1' >> artifacts/queue/jobs.txt
#   start runner:  nohup scripts/jobq.sh > artifacts/queue/runner.log 2>&1 &
# The runner pops the first line, runs it, records it in done.txt (exit code), and repeats until jobs.txt is empty.
# A job interrupted by a crash is still in current.txt: re-append it by hand after checking its output.
cd "$(dirname "$0")/.."
Q=artifacts/queue
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.45 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35
if [ -f $Q/runner.pid ] && kill -0 "$(cat $Q/runner.pid)" 2>/dev/null; then echo "runner already running"; exit 0; fi
echo $$ > $Q/runner.pid
while true; do
  [ -s $Q/jobs.txt ] || break
  job=$(head -n 1 $Q/jobs.txt)
  tail -n +2 $Q/jobs.txt > $Q/jobs.tmp && mv $Q/jobs.tmp $Q/jobs.txt
  echo "$job" > $Q/current.txt
  echo "[$(date '+%F %T')] START $job"
  bash -c "$job"; rc=$?
  echo "[$(date '+%F %T')] rc=$rc $job" | tee -a $Q/done.txt
  rm -f $Q/current.txt
done
rm -f $Q/runner.pid
echo "[$(date '+%F %T')] queue empty"
