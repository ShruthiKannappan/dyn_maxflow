#!/bin/bash

g++ create_update_bin.cpp -o cb_update

# Directories to process
dirs=("inc" "dec" "mixed")

for d in "${dirs[@]}"; do
    dir="../data/updates/$d"

    if [ -d "$dir" ]; then
        for txt in "$dir"/*.txt; do
            [ -e "$txt" ] || continue  # skip if no .txt files
            base=$(basename "$txt" .txt)
            bin="$dir/$base.bin"
            ./cb_update "$txt" "$bin"
        done
    fi
done
