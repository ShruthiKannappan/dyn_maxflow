// FOR BC: nvcc bc_dsl_v2.cu -arch=sm_60 -std=c++14 -rdc=true # HW must support CC 6.0+ Pascal or after
#ifndef GENCPP_DYN_TOPO_H
#define GENCPP_DYN_TOPO_H
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>
#include "graph.hpp"
#include <cooperative_groups.h>
#include <string>
#include <sstream>
#include <iostream>
#include<cstring>


void staticMaxFlow(graph& g,int source0,int sink0,int* d_residual_capacity,
  int* d_rev_residual_capacity,int* d_reverse_edge,int* d_excess,int* d_parallel_edge,
  int kernel_cycles0);





__global__ void staticMaxFlow_kernel_1(int V,  int* d_meta, int* d_data, int* d_weight,int* d_residual_capacity,int* d_rev_residual_capacity,int* d_parallel_edge, int *d_excess){ 
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) { // FOR NBR ITR 
    int e1 = edge;
    d_residual_capacity[e1] = d_weight[e1];
    int p = d_parallel_edge[e1];
    d_rev_residual_capacity[e1] = d_weight[p];
  }
  d_excess[v] = 0;
}



__global__ void staticMaxFlow_kernel_7(int source0,int V, int E, int* d_meta, int* d_data, int* d_residual_capacity,int* d_excess,int* d_parallel_edge,int* d_rev_residual_capacity){ // BEGIN KER FUN via ADDKERNEL
  unsigned v = source0;

    for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) { 
      int vv = d_data[edge];
      int forward_edge = edge;
      int d = d_residual_capacity[forward_edge];
      if (d > 0){
        d_excess[vv] += d;
        d_excess[v] -= d;
        d_residual_capacity[forward_edge] -= d;
        d_rev_residual_capacity[forward_edge] += d;
        int p = d_parallel_edge[forward_edge]; 
        d_residual_capacity[p] += d;
        d_rev_residual_capacity[p] -= d;
      } 
    } 
}

__device__ bool flag1 ; // DEVICE ASSTMENT in .h


__global__ void staticMaxFlow_kernel_10(int source0,int sink0,int V, int E,int* d_excess,int* d_level, int *d_height){ // BEGIN KER FUN via ADDKERNEL
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  d_height[v] = V;
  d_level[v] = -1;
  if (v != source0){ // if filter begin 
    if (d_excess[v] < 0 || v == sink0){ // if filter begin 
      d_level[v] = 0;

    }
  } 
}
__device__ bool bfsflag ;

__global__ void staticMaxFlow_kernel_11(int hops,int V, int E, int* d_meta, int* d_data,int* d_residual_capacity,int* d_level,int* d_rev_residual_capacity,int* d_height){ 
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  if (d_level[v] == hops){
    int curhops = hops;
    d_height[v] = curhops;
    for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) { 
      int w = d_data[edge];
      int e = edge; 
      if (d_rev_residual_capacity[e] > 0 && d_level[w] == -1){ 
        d_level[w] = curhops + 1;
        bfsflag = true;
      }
    }
  }
}
__global__ void staticMaxFlow_kernel_14(int source0,int sink0,int kernel_cycles0,int V, int E, int* d_meta, int* d_data, int* d_excess,int* d_parallel_edge,int* d_rev_residual_capacity,int* d_residual_capacity,int* d_height){ // BEGIN KER FUN via ADDKERNEL
  float num_nodes  = V;
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  if (d_excess[v] > 0 && v != source0 && v != sink0 && d_height[v] < num_nodes){ // if filter begin 
    flag1 = true;
    int cycle = kernel_cycles0;
    do{
      if (d_excess[v] > 0 && d_height[v] < V){ 
        int ex1 = d_excess[v];
        int hh = INT_MAX; 
        int v_0 = -1;
        int forward_edge = -1;
        for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) {
          int vv = d_data[edge];
          int e = edge;
          int h1 = d_height[vv];
          if (h1 < hh && d_residual_capacity[e] > 0){
            v_0 = vv;
            hh = h1;
            forward_edge = e;
          }
        }
        if (d_height[v] > hh && v_0 != -1){
            int fec = d_residual_capacity[forward_edge];
            int p = d_parallel_edge[forward_edge];
            int d = fec;
            d = min(ex1,fec);
            atomicSub(& d_excess[v] , d);
            atomicAdd(& d_excess[v_0] , d);
            atomicSub(& d_residual_capacity[forward_edge] , d);
            atomicAdd(& d_rev_residual_capacity[forward_edge] , d);
            atomicAdd(& d_residual_capacity[p] , d);
            atomicSub(& d_rev_residual_capacity[p] , d);
        }
        else {
          if (v_0 != -1){
            d_height[v] = hh + 1;
          }
        }
      }
      else break;
      cycle = cycle - 1;
    }while(cycle > 0);

  } // if filter end
} // end KER FUNC
__global__ void staticMaxFlow_kernel_17(int source0,int sink0,int V, int E, int* d_meta, int* d_data,int* d_rev_residual_capacity,int* d_parallel_edge,int* d_residual_capacity,int* d_excess,int* d_height){ // BEGIN KER FUN via ADDKERNEL
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
    int h = d_height[v];
    for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) { 
      int vv = d_data[edge];
      int e = edge;
      if (d_rev_residual_capacity[e] > 0){ 
        if (d_height[vv] > h+1){
          int d = d_rev_residual_capacity[e]; 
          d_rev_residual_capacity[e] = d_rev_residual_capacity[e] - d;
          d_residual_capacity[e] = d_residual_capacity[e] + d;
          int p = d_parallel_edge[e];
          d_rev_residual_capacity[p] = d_rev_residual_capacity[p] + d;
          d_residual_capacity[p] = d_residual_capacity[p] - d;
          atomicSub(& d_excess[vv] , d);
          atomicAdd(& d_excess[v] , d);
        } 
      }
    }
}


__global__ void incremental_kernel_20(int V, int E, int* d_meta, int* d_data,int* d_rev_residual_capacity,int* d_residual_capacity){ // BEGIN KER FUN via ADDKERNEL
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) { // FOR NBR ITR 
    int e = edge;
    if (d_residual_capacity[e] < 0){ // if filter begin 
      d_rev_residual_capacity[e] = d_rev_residual_capacity[e] + d_residual_capacity[e];
      d_residual_capacity[e] = 0;

    } // if filter end
    if (d_rev_residual_capacity[e] < 0){ // if filter begin 
        d_residual_capacity[e] = d_residual_capacity[e] + d_rev_residual_capacity[e];
        d_rev_residual_capacity[e] = 0;
      }
  }
}
__global__ void incremental_kernel_22(int V, int E, int* d_meta, int* d_weight,int* d_residual_capacity,int* d_excess){ // BEGIN KER FUN via ADDKERNEL
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  int e1 = 0; // DEVICE ASSTMENT in .h

  for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) { // FOR NBR ITR 
    int e = edge;
    e1 = e1 + d_residual_capacity[e]-d_weight[e];
  } 
  d_excess[v] = e1; //InitIndex
} // end KER FUNC


__global__ void OnAdd_kernel(update* d_updateBatch, int batchelements, int* d_meta, int* d_data, int* d_weight,int* d_rev_residual_capacity,int* d_parallel_edge,int* d_residual_capacity){ // BEGIN KER FUN via ADDKERNEL
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= batchelements) return;
  update u = d_updateBatch[tid];
  if( u.type!='a') return;
  int uedge = -1; 
  int src = u.source; 
  int dest = u.destination; 
  if(src==dest) return;
  int new_capacity = u.weight;
  for (int edge = d_meta[src]; edge < d_meta[src+1]; edge++) { 
    int dd = d_data[edge];
    if (dd == dest){ 
      uedge = edge;
    } 
  } 
  d_residual_capacity[uedge] = d_residual_capacity[uedge] + new_capacity - d_weight[uedge];
  int p = d_parallel_edge[uedge];
  d_rev_residual_capacity[p] = d_rev_residual_capacity[p] + new_capacity - d_weight[uedge];
  d_weight[uedge] = new_capacity;
} // end KER FUNC

__global__ void recalculate_max_flow_kernel_39(int source,int sink,int V, int E,int* d_excess){ // BEGIN KER FUN via ADDKERNEL
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  if (d_excess[v] < 0 && v != source){ // if filter begin 
    atomicAdd(& d_excess[sink] , d_excess[v]);

  } // if filter end
} // end KER FUNC

#endif
