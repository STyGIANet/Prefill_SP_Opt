#!/bin/bash

# --eligible-max-tokens 490000 # This is a bit annoying. Can we get rid of this? Kv cache reservation and vram capacity is the issue.
./run_prefill_experiments.sh --suite full --datasets pg19 longbench --schedulers laps tetris spsrpt --offered-load 25000 --spsrpt-quantum 1.0 --short-threshold 8192 --eligible-max-tokens 490000 --recalibrate "$@"
