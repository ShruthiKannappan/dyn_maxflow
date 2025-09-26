# Dataset Preparation for Dynamic Max-Flow

This directory contains scripts to prepare graph inputs and update batches for experiments in our VLDB paper.  
The steps below describe how to generate weighted graphs, binary input files, and update sets.

---

## 📂 Directory Structure

We use the following structure to store the inputs under `data/`:

```

data/
├── raw_dataset/          # Original graphs in Matrix Market (.mtx) format
|   └── johnson8-2-4.mtx
├── processed_dataset/    # Processed graphs and updates
└──updates/             
    ├── inc/              # Incremental updates
    ├── dec/              # Decremental updates
    └── mixed/            # Mixed updates

````

---

## Step 1: Create a weighted Graph

Convert a raw `.mtx` graph into a weighted `.txt` edge-list format:

```bash
g++ weight.cpp -o weightg
./weightg <unweighted_graph_path> <weighted_graph_path>
````
Eg. 
```bash
g++ weight.cpp -o weightg
./weightg ../dataset/raw_dataset/johnson8-2-4.mtx \  
../dataset/dataset_with_updates/johnson.txt
````

Convert the graph to a bi-directional CSR and store in a binary.
```bash
g++ create_bin.cpp -o cb
./cb <weighted_graph_path> <weight_graph_bin_path>
```
Eg. 

```bash

./cb ../dataset/dataset_with_updates/johnson.txt \
     ../dataset/dataset_with_updates/johnson.bin
```

---

## Step 2: Identify Source and Sink

Each graph requires a **source** and **sink** node for the max-flow computation.
You may determine them manually or using helper utilities. 
In the following dataset and example we can use 
```
Source: 25
Sink:   2
```

---

## Step 3: Generate Update Batches

Use the `create_txt_updates.sh` script to generate incremental, decremental, and mixed updates:

```bash
bash create_txt_updates.sh <weighted_graph_name> <update_prefix> <source> <sink>
```

Eg.
```bash
bash create_txt_updates.sh johnson.txt update_john 25 2
```
Output: 10 `.txt` update files under `inc/`, `dec/`, or `mixed/` where ith file is of size i%|E|

Convert all update `.txt` files into `.bin` format for GPU execution:

```bash
bash conv_bin.sh
```

This scans all update directories (`inc/`, `dec/`, `mixed/`) and generates `.bin` counterparts.

---
