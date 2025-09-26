# Makefile for building dynamic max-flow benchmarks

NVCC = nvcc


all: dyn_data_bench_large dyn_data_bench dyn_pp_bench_large dyn_pp_bench dyn_topo_bench_large dyn_topo_bench

dyn_data_bench_large: dyn_data_bench_large.cu
	$(NVCC) $< -o $@

dyn_data_bench: dyn_data_bench.cu
	$(NVCC) $< -o $@

dyn_pp_bench_large: dyn_pp_bench_large.cu
	$(NVCC) $< -o $@

dyn_pp_bench: dyn_pp_bench.cu
	$(NVCC) $< -o $@

dyn_topo_bench_large: dyn_topo_bench_large.cu
	$(NVCC) $< -o $@

dyn_topo_bench: dyn_topo_bench.cu
	$(NVCC) $< -o $@


clean:
	rm -f dyn_data_bench_large dyn_data_bench dyn_pp_bench_large dyn_pp_bench dyn_topo_bench_large dyn_topo_bench
