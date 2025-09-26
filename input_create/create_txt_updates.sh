#!/bin/bash

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <weighted_graph> <outputprefix> [src] [sink]"
    exit 1
fi

g++ update_gen.cpp -o ug
g++ cut_p.cpp -o p

weighted_graph=$1
output_prefix=$2
src=${3:-0}   # default src=0 if not provided
sink=${4:-0}  # default sink=0 if not provided

# Loop over all update types
for type in dec inc mixed; do
    if [[ "$type" == "dec" ]]; then
        inc_p=0
    elif [[ "$type" == "mixed" ]]; then
        inc_p=50
    elif [[ "$type" == "inc" ]]; then
        inc_p=100
    fi

    echo "Processing type: $type"

    # Generate long update file
    batch_size=$(./ug "../data/processed_dataset/$weighted_graph" \
                       "../data/updates/$type/long_update.txt" \
                       11 "$inc_p" "$src" "$sink" \
                 | grep "batch size:" | awk '{print $8}')

    max_batch_size=$((batch_size * 10))

    # Create trimmed update file
    shuf "../data/updates/$type/long_update.txt" \
        | head -n "$max_batch_size" \
        > "../data/updates/$type/trim_update.txt"

    # Split into batches
    for i in {1..10}; do
        ./p "../data/updates/$type/trim_update.txt" \
             "../data/updates/$type/$output_prefix" \
             "$batch_size" "$i"
    done

    # Clean up temporary files
    rm "../data/updates/$type/trim_update.txt" "../data/updates/$type/long_update.txt"
done
