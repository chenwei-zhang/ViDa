
#!/bin/bash


parallel -j 32 python3 Machinek_trajmode_parallel.py ::: {161..199}
echo "All processes have completed."