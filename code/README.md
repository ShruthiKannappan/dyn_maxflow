# Code Directory for Dynamic Max-Flow Benchmarks

This directory contains the CUDA implementations and benchmarking scripts used in our VLDB paper.  
The `Makefile` builds all binaries from the `.cu` source files.

---

## Build

Simply run:

```bash
make
````

This will compile all `.cu` files and generate binaries with the same base file names.

---

## File Overview

### Usage

* **`*_bench.cu` binaries**: runs the update batches (1–10%) separately; each update file is applied on the original graph. Useful when there are many small update files.
* **`*_bench_large.cu` binaries**: runs the update batch on the original graph as a single file. Use these when the update **batch is large**.

Binaries are generated with the same base name as the `.cu` files. For example:

```bash
nvcc dyn_data_bench.cu -o dyn_data_bench
nvcc dyn_data_bench_large.cu -o dyn_data_bench_large
```

---

## Summary of Algorithms

| Algorithm | Description                                                       |
| --------- | ----------------------------------------------------------------- |
| `topo`    | Dynamic Push-Relabel Algorithm with **topology-based** processing |
| `data`    | Dynamic Push-Relabel Algorithm with **data-based** processing     |
| `pp`      | Dynamic Push-Pull Algorithm with **data-based** processing        |

---

