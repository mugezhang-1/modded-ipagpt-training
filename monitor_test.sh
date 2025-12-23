#!/bin/bash
JOB_ID=4551583
MAX_WAIT=60

echo "Monitoring job $JOB_ID..."
for i in $(seq 1 $MAX_WAIT); do
    sleep 5
    status=$(squeue -j $JOB_ID 2>/dev/null | tail -n +2)

    if [ -z "$status" ]; then
        echo "Job completed! Checking output..."
        sleep 2
        if [ -f "test-checkpoint-loading-${JOB_ID}.out" ]; then
            cat "test-checkpoint-loading-${JOB_ID}.out"
        else
            echo "Output file not found yet, waiting..."
            sleep 5
            cat "test-checkpoint-loading-${JOB_ID}.out" 2>/dev/null || echo "Still no output file"
        fi
        break
    else
        echo "[$i/60] Job status: $(echo $status | awk '{print $5}')"
    fi

    if [ $i -eq $MAX_WAIT ]; then
        echo "Timeout - job still running. Current status:"
        squeue -j $JOB_ID
        echo "Check output later with: cat test-checkpoint-loading-${JOB_ID}.out"
    fi
done
