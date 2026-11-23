#!/bin/bash

./run_prefill_experiments.sh --suite full --datasets pg19 longbench --schedulers laps tetris spsrpt --offered-load 25000 --spsrpt-quantum 1.0 --short-threshold 8192 --recalibrate
