// FOR BC: nvcc bc_dsl_v2.cu -arch=sm_60 -std=c++14 -rdc=true # HW must support CC 6.0+ Pascal or after
#ifndef GENCPP_DYN_PP_H
#define GENCPP_DYN_PP_H
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
#define BLOCKSIZE 1024
#define FULL_MASK 0xFFFFFFFF



__device__ bool bfsflag ; 





__global__ void init_kernel(int V,  int* d_meta, int* d_data, int* d_weight,int* d_residual_capacity,int* d_rev_residual_capacity,int* d_parallel_edge, int *d_excess){ 
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



__global__ void staticMaxFlow_kernel_7(int source0, int sink0,int* d_meta, int* d_data, int* d_residual_capacity,int* d_excess,int* d_parallel_edge,int* d_rev_residual_capacity){ // BEGIN KER FUN via ADDKERNEL
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

   v = sink0;

    for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) { 
      int vv = d_data[edge];
      int d = d_rev_residual_capacity[edge];
      if (d > 0){
        d_excess[v] += d;
        d_excess[vv] -= d;
        d_residual_capacity[edge] += d;
        d_rev_residual_capacity[edge] -= d;
        int p = d_parallel_edge[edge]; 
        d_residual_capacity[p] -= d;
        d_rev_residual_capacity[p] += d;
      } 
    } 

}



__global__ void pp_bfs_init(int source, int sink,int V, int *d_excess, bool*d_partB, int *d_level,int *d_height){
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V ) return;
  d_level[v] = -1;
  d_height[v] = V;
  if(v==source || v==sink) d_level[v] = 0;
  if((d_partB[v] &&  (d_excess[v]<0)) || ((!d_partB[v])  && (d_excess[v]>0))) d_level[v] = 0;
}


__global__ void final_bfs_init(int *d_part, int nodes,int source, int sink,int V,int *d_excess, bool*d_partB, int *d_level,int *d_height){
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= nodes) return;
  int v = d_part[tid];
  d_level[v] = -1;
  d_height[v] = V;
  if(d_excess[v]<0)d_level[v] = 0;
}





__global__ void pp_bfs(int source, int sink,int hops,int V,int* d_meta, int* d_data,int* d_level,int *d_residual_capacity,int* d_rev_residual_capacity,int* d_height, bool *d_partB){ // BEGIN KER FUN via ADDKERNEL
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  bool push = d_partB[v];
  if (d_level[v] == hops){  
    int curhops = hops; 
    d_height[v] = curhops;
    for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) { 
      int w = d_data[edge];
      int e = edge; 
      if(w==source|| w==sink) continue;
      if ( push && d_rev_residual_capacity[e] > 0 && d_level[w] == -1){ 
          d_level[w] = curhops + 1;
          bfsflag = true;
      } 
      if ((!push) && d_residual_capacity[e] > 0 && d_level[w] == -1 ){
          d_level[w] = curhops + 1;
          bfsflag = true;
      }
    } 
  }  
} 


__global__ void final_bfs(int *d_part, int nodes,int source, int sink,int hops,int V,int* d_meta, int* d_data,int* d_level,int *d_residual_capacity,int* d_rev_residual_capacity,int* d_height, bool *d_partB){ // BEGIN KER FUN via ADDKERNEL
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= nodes) return;
  int v = d_part[tid];
  if (d_level[v] == hops){  
    int curhops = hops; 
    d_height[v] = curhops;
    for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) { 
      int w = d_data[edge];
      if(d_partB[w]) continue;
      int e = edge; 
      if (d_rev_residual_capacity[e] > 0 && d_level[w] == -1){ 
          d_level[w] = curhops + 1;
          bfsflag = true;
      } 
    } 
  }  
} 




__global__ void push_kernel(int kernel_cycles,int V,  int* d_meta, int* d_data, int* d_rev_residual_capacity,int* d_excess,int* d_parallel_edge,int* d_residual_capacity,int* d_height, int *d_set_ptr,int *d_set){ // BEGIN KER FUN via ADDKERNEL
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= d_set_ptr[0]) return;
  int v = d_set[tid];
  int cycle = kernel_cycles; 
  while(cycle--){
    int ex1 = d_excess[v];
    if (ex1 > 0 && d_height[v] < V){ 
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
    } else break;
  }
} // end KER FUNC





__global__ void pp_push_kernel(int kernel_cycles,int V,  int* d_meta, int* d_data, int* d_rev_residual_capacity,int* d_excess,int* d_parallel_edge,int* d_residual_capacity,int* d_height, int *d_set_ptr,int *d_set, bool *d_partB){ // BEGIN KER FUN via ADDKERNEL
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= d_set_ptr[0]) return;
  int v = d_set[tid];
  int cycle = kernel_cycles; 
  while(cycle--){
    int ex1 = d_excess[v];
    if (ex1 > 0 && d_height[v] < V){ 
      int hh = INT_MAX; 
      int v_0 = -1;
      int forward_edge = -1;
      for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) { 
        int vv = d_data[edge];
        if(!d_partB[vv]) continue;
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
    } else break;
  }
} // end KER FUNC




__global__ void pull_kernel(int kernel_cycles,int V, int* d_meta, int* d_data, int* d_rev_residual_capacity,int* d_excess,int* d_parallel_edge,int* d_residual_capacity,int* d_height, int *d_set_ptr,int *d_set, bool *d_partB){ // BEGIN KER FUN via ADDKERNEL
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= d_set_ptr[0]) return;
  int v = d_set[tid];
  int cycle = kernel_cycles; 
  while(cycle--){
    int ex1 = d_excess[v]*(-1) ;
    if (ex1 > 0){ 
      int hh = INT_MAX; 
      int v_0 = -1;
      int forward_edge = -1;
      for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) { 
        int vv = d_data[edge];
        if(d_partB[vv]) continue;
        int e = edge;
        
        int h1 = d_height[vv]; 
        if (h1 < hh && d_rev_residual_capacity[e] > 0){ 
          v_0 = vv;
          hh = h1;
          forward_edge = e;
        } 
      }
      if (d_height[v] > hh && v_0 != -1){  
        int rec = d_rev_residual_capacity[forward_edge]; 
        int p = d_parallel_edge[forward_edge]; 
        int d = rec; 
          d = min(ex1,rec);
        atomicSub(& d_excess[v_0] , d);
        atomicAdd(& d_excess[v] , d);
        atomicSub(& d_rev_residual_capacity[forward_edge] , d);
        atomicAdd(& d_residual_capacity[forward_edge] , d);
        atomicAdd(& d_rev_residual_capacity[p] , d);
        atomicSub(& d_residual_capacity[p] , d);
      } 
      else {
        if (v_0 != -1){  
          d_height[v] = hh + 1;
        }  
      }
    }
    else break;
  }
} // end KER FUNC

__global__ void remove_push_final( int* d_meta, int* d_data,int* d_rev_residual_capacity,int* d_parallel_edge,int* d_residual_capacity,int* d_excess,int* d_height, int *d_set_ptr, int *d_set, bool *d_partfinal){ // BEGIN KER FUN via ADDKERNEL
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= d_set_ptr[0]) return;
  int v = d_set[tid];
  int h = d_height[v]; 
  for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) {
    int vv = d_data[edge];
    if(!d_partfinal[vv]) continue;
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


__global__ void remove_push( int* d_meta, int* d_data,int* d_rev_residual_capacity,int* d_parallel_edge,int* d_residual_capacity,int* d_excess,int* d_height, int *d_set_ptr, int *d_set){ // BEGIN KER FUN via ADDKERNEL
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= d_set_ptr[0]) return;
  int v = d_set[tid];
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

__global__ void find_part(int V, int *d_level,int *d_height, bool *d_partB, int *d_set, int *d_set_ptr, bool *d_partfinal){
    unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
    bool isActive = false;
  unsigned mask;
  int windex = -1;
  int index = -1;
  int actives =  0;
  unsigned int laneId = threadIdx.x % 32;
  if(v<V){
    d_partfinal[v] = false;
    if(d_level[v]==-1){
      isActive = true;
      d_partB[v] = false;
      d_partfinal[v] = true;
    }
    d_level[v] = -1;
    d_height[v] = V;
  }

    mask = __ballot_sync(FULL_MASK,isActive);
  int myVoteRank = __popc(mask & ((1 << laneId) - 1));
  actives = __popc(mask);
  if(laneId==0 && actives>0) { index = atomicAdd(d_set_ptr,actives);}
  windex = __shfl_sync(FULL_MASK, index, 0);
  int my_index = myVoteRank+windex;
  if(isActive) {d_set[my_index] = v;  } 
}

__global__ void pp_remove_push( int* d_meta, int* d_data,int* d_rev_residual_capacity,int* d_parallel_edge,int* d_residual_capacity,int* d_excess,int* d_height, int *d_set_ptr, int *d_set, bool *d_partB){ // BEGIN KER FUN via ADDKERNEL
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= d_set_ptr[0]) return;
  int v = d_set[tid];
  int h = d_height[v]; 
  for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) {
    int vv = d_data[edge];
    if(!d_partB[vv]) continue;
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
__global__ void remove_pull(int* d_meta, int* d_data,int* d_rev_residual_capacity,int* d_parallel_edge,int* d_residual_capacity,int* d_excess,int* d_height, int *d_set_ptr, int *d_set, bool *d_partB){ // BEGIN KER FUN via ADDKERNEL
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= d_set_ptr[0]) return;
  int v = d_set[tid];
  int h = d_height[v]; 
  for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) {
    int vv = d_data[edge];
    if(d_partB[vv]) continue;
    int e = edge;
    if (d_residual_capacity[e] > 0){ 
      if (d_height[vv] > h+1){
        int d = d_residual_capacity[e]; 
        d_rev_residual_capacity[e] = d_rev_residual_capacity[e] + d;
        d_residual_capacity[e] = d_residual_capacity[e] - d;
        int p = d_parallel_edge[e];
        d_rev_residual_capacity[p] = d_rev_residual_capacity[p] - d;
        d_residual_capacity[p] = d_residual_capacity[p] + d;
        atomicSub(& d_excess[v] , d);
        atomicAdd(& d_excess[vv] , d);
      } 
    }
  } 
} 

__global__ void incremental_kernel_20(int V,  int* d_meta, int* d_data,int* d_rev_residual_capacity,int* d_residual_capacity, bool *d_partB, int *d_excess, int *d_weight){ // BEGIN KER FUN via ADDKERNEL

  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  bool self = d_partB[v];
  int e1 = 0;
  for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) { 
    int e = edge;
    int nbr = d_data[e];
    if (d_residual_capacity[e] < 0){  
      d_rev_residual_capacity[e] = d_rev_residual_capacity[e] + d_residual_capacity[e];
      d_residual_capacity[e] = 0;
    } 
    if (d_rev_residual_capacity[e] < 0){ 
        d_residual_capacity[e] = d_residual_capacity[e] + d_rev_residual_capacity[e];
        d_rev_residual_capacity[e] = 0;
    }
    if(self!=d_partB[nbr]){
        if(self){
          int d = d_rev_residual_capacity[e];
          d_rev_residual_capacity[e]-=d;
          d_residual_capacity[e]+=d;
        } else {
          int d = d_residual_capacity[e];
          d_residual_capacity[e]-=d;
          d_rev_residual_capacity[e]+=d;
        }
    }
    e1 = e1 + d_residual_capacity[e]-d_weight[e];
  }
  d_excess[v] = e1;
}



__global__ void find_cut(int V,int *d_level , bool *d_partB){
    unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  if(d_level[v]==-1){
    d_partB[v] = false;
  } else {
    d_partB[v] = true;
  }

}


__global__ void find_final_cut(int V,int *d_level , bool *d_partB){
    unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  if( !d_partB[v]){
    if(d_level[v]!=-1){
      d_partB[v] = true;
    }
  }
}
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






__global__ void create_pp_wl(int source, int sink,int *d_excess, int *d_set_ptr,int *d_set2_ptr,int *d_set,int *d_set2,int *d_height, int V, bool *d_partB){
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  
    bool isActivePull = false;
  unsigned maskPull;
  int windexPull = -1;
  int indexPull = -1;
  int activesPull =  0;

  unsigned int laneId = threadIdx.x % 32;
        bool isActivePush = false;
  unsigned maskPush;
  int windexPush = -1;
  int indexPush = -1;
  int activesPush =  0;
  if(v<V  && v!=sink && v!=source ){
    if(d_partB[v]){
      if(  d_excess[v]>0 && d_height[v]<V ){
        isActivePush = true;
      }
    } else{
      if(d_excess[v]<0 && d_height[v]<V ){
        isActivePull = true;
      }
    }
 
  }

  maskPull = __ballot_sync(FULL_MASK,isActivePull);
  int myVoteRankPull = __popc(maskPull & ((1 << laneId) - 1));
  activesPull = __popc(maskPull);
  if(laneId==0 && activesPull>0) { indexPull = atomicAdd(d_set2_ptr,activesPull);}
  windexPull = __shfl_sync(FULL_MASK, indexPull, 0);
  int my_indexPull = myVoteRankPull+windexPull;
  if(isActivePull) {d_set2[my_indexPull] = v;  } 


    maskPush = __ballot_sync(FULL_MASK,isActivePush);
  int myVoteRankPush = __popc(maskPush & ((1 << laneId) - 1));
  activesPush = __popc(maskPush);
  if(laneId==0 && activesPush>0) { indexPush = atomicAdd(d_set_ptr,activesPush);}
  windexPush = __shfl_sync(FULL_MASK, indexPush, 0);
  int my_indexPush = myVoteRankPush+windexPush;
  if(isActivePush) {d_set[my_indexPush] = v;  } 


}





__global__ void my_init_1(int V, int *d_height, int *d_level,int* d_excess,int source, int sink){
  int v = blockIdx.x * blockDim.x + threadIdx.x; 
  if(v>=V) return;
  d_height[v] = V;
  d_level[v] = -1;
  if(v!=source && d_excess[v]<0 )d_level[v] = 0;
  if(v==sink) d_level[v] = 0;
}

__global__ void staticMaxFlow_kernel_11(int hops,int V,int* d_meta, int* d_data,int* d_level,int* d_rev_residual_capacity,int* d_height){ // BEGIN KER FUN via ADDKERNEL
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

__global__ void staticMaxFlow_kernel_12(int hops,int V,int* d_meta, int* d_data,int* d_level,int* d_residual_capacity,int* d_height){ // BEGIN KER FUN via ADDKERNEL
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  if (d_level[v] == hops){  
    int curhops = hops; 
    d_height[v] = curhops;
    for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) { 
      int w = d_data[edge];
      int e = edge; 
      if (d_residual_capacity[e] > 0 && d_level[w] == -1){ 
        d_level[w] = curhops + 1;
        bfsflag = true;
      } 
    } 
  } 
} 



__global__ void create_wl(int source, int sink,int *d_excess, int V,  int *d_set_ptr,int *d_height,int *d_set){
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
    bool isActive = false;
  unsigned mask;
  int windex = -1;
  int index = -1;
  int actives =  0;
  unsigned int laneId = threadIdx.x % 32;
  if(v<V){
    if(d_excess[v]>0 && d_height[v]<V && v!=source && v!=sink){
      isActive = true;
    }
  }

    mask = __ballot_sync(FULL_MASK,isActive);
  int myVoteRank = __popc(mask & ((1 << laneId) - 1));
  actives = __popc(mask);
  if(laneId==0 && actives>0) { index = atomicAdd(d_set_ptr,actives);}
  windex = __shfl_sync(FULL_MASK, index, 0);
  int my_index = myVoteRank+windex;
  if(isActive) {d_set[my_index] = v;  } 
}

__global__ void create_final_wl(int *d_part, int nodes,int source, int sink,int *d_excess, int V,  int *d_set_ptr,int *d_height,int *d_set){
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= nodes) return;
  int v = d_part[tid];
    bool isActive = false;
  unsigned mask;
  int windex = -1;
  int index = -1;
  int actives =  0;
  unsigned int laneId = threadIdx.x % 32;
 
    if(d_excess[v]>0 && d_height[v]<V){
      isActive = true;
    }
  

    mask = __ballot_sync(FULL_MASK,isActive);
  int myVoteRank = __popc(mask & ((1 << laneId) - 1));
  actives = __popc(mask);
  if(laneId==0 && actives>0) { index = atomicAdd(d_set_ptr,actives);}
  windex = __shfl_sync(FULL_MASK, index, 0);
  int my_index = myVoteRank+windex;
  if(isActive) {d_set[my_index] = v;  } 
}

#endif