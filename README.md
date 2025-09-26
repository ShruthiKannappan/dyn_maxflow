# Dynamic Max-Flow on GPUs 

This repository contains the code, datasets, and scripts used in our paper on **Efficient Dynamic MaxFlow Computation on GPUs**.

---

## Repository Structure

The repository is organized into three main directories:

| Directory      | Description                                                                                     |
|----------------|-------------------------------------------------------------------------------------------------|
| `data/`        | Contains the graph datasets and update files (both `.txt` and `.bin` formats).                 |
| `input_create/`| Contains scripts to create weighted graphs, generate update files, and convert them to binary. |
| `code/`        | Contains CUDA implementations of the algorithms discussed in the paper and benchmarking scripts.|

---

## Getting Started

1. **Prepare Inputs**

   Navigate to the `input_create/` directory and follow the instructions in its README.  

   - This will generate the **weighted graph** and **update binaries** required for running the algorithms.
   - Ensure that the **original graph binary** and **update binaries** are ready before proceeding.
````

2. **Run CUDA Implementations**

   Once the binaries are ready, move to the `code/` directory:

   * Follow the instructions in the `code/README.md`.
   * Choose the algorithm implementation and the dataset you want to benchmark.
   * Run the appropriate binary to obtain results for your experiments.

---

## Notes

* The **input_create** folder prepares the inputs in the format expected by the CUDA programs.
* The **code** folder contains the actual GPU algorithms; the binaries are generated via `make`.
* The benchmarking procedure (1–10% updates or single large batch) is described in the `code/README.md`.

---

## ✅ Citation

If you use this repository or datasets in your work, please cite our paper:

```
@inproceedings{YourPaperKey,
  author    = {Your Name and Co-Authors},
  title     = {Dynamic Max-Flow on GPUs},
  booktitle = {Proceedings of the VLDB Endowment (VLDB)},
  year      = {2025}
}
```


