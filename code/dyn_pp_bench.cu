// FOR BC: nvcc bc_dsl_v2.cu -arch=sm_60 -std=c++14 -rdc=true # HW must support CC 6.0+ Pascal or after
#include "dyn_pp.h"
const int MAX_FILENAME_LEN = 1024;
#define THREADS_PER_BLOCK 1024

void part_pp(int source, int sink,int *d_level, int *d_height, int *d_meta, int *d_data, int *d_residual_capacity, 
int *d_rev_residual_capacity, bool *d_partB, int *d_excess, int V, 
int *d_set, int *d_set_ptr, int *d_set2, int *d_set2_ptr,
int kernel_cycles, int *d_parallel_edge)
{

// perform bfs for the two streams
// create two streams and push the kernels in the respective streams and wait for termination.
// perform ballot based itself here


   unsigned numThreads   = (V< THREADS_PER_BLOCK)?V: THREADS_PER_BLOCK;
  unsigned numBlocks    = (V+numThreads-1)/numThreads;

  
  bool bfsflag = false;
  pp_bfs_init<<<numBlocks, numThreads>>>(source,sink,V,d_excess, d_partB,d_level,d_height);
  int hops = 0;
  do{
    bfsflag = false;
    cudaMemcpyToSymbol(::bfsflag, &bfsflag, sizeof(bool), 0, cudaMemcpyHostToDevice);
    pp_bfs<<<numBlocks, numThreads>>>(source,sink,hops,V,d_meta,d_data,d_level,d_residual_capacity,d_rev_residual_capacity,d_height,d_partB);
    cudaDeviceSynchronize();
    hops++;
    cudaMemcpyFromSymbol(&bfsflag, ::bfsflag, sizeof(bool), 0, cudaMemcpyDeviceToHost);
  }while(bfsflag);
cudaDeviceSynchronize();
    cudaMemset(d_set_ptr,0,sizeof(int));
    cudaMemset(d_set2_ptr,0,sizeof(int));
create_pp_wl<<<numBlocks, numThreads>>>(source,sink,d_excess,d_set_ptr,d_set2_ptr,d_set,d_set2,d_height, V, d_partB);
cudaDeviceSynchronize();
    int active = 0;
    cudaMemcpy(&active,d_set_ptr,sizeof( int),cudaMemcpyDeviceToHost);

    int active2 = 0;
    cudaMemcpy(&active2,d_set2_ptr,sizeof( int),cudaMemcpyDeviceToHost);
cudaStream_t push,pull;
cudaStreamCreate(&push);
cudaStreamCreate(&pull);
int cnt = 0;
while(active+active2>0){
  cnt++;
  if(active>0){
        unsigned activenumThreads   = (active < THREADS_PER_BLOCK)? active: THREADS_PER_BLOCK;
    unsigned activenumBlocks    = (active+activenumThreads-1)/activenumThreads;
    pp_push_kernel<<<activenumBlocks,activenumThreads,0,push>>>(kernel_cycles,V,d_meta,d_data,d_rev_residual_capacity,d_excess,d_parallel_edge,d_residual_capacity,d_height,d_set_ptr,d_set,d_partB);
    pp_remove_push<<<activenumBlocks,activenumThreads,0,push>>>(d_meta,d_data,d_rev_residual_capacity,d_parallel_edge,d_residual_capacity,d_excess,d_height,d_set_ptr,d_set,d_partB);
  }

  if(active2>0){
    unsigned active2numThreads   = (active2 < THREADS_PER_BLOCK)? active2: THREADS_PER_BLOCK;
    unsigned active2numBlocks    = (active2+active2numThreads-1)/active2numThreads;
    pull_kernel<<<active2numBlocks,active2numThreads,0,pull>>>(kernel_cycles,V,d_meta,d_data,d_rev_residual_capacity,d_excess,d_parallel_edge,d_residual_capacity,d_height,d_set2_ptr,d_set2,d_partB);
    remove_pull<<<active2numBlocks,active2numThreads,0,pull>>>(d_meta,d_data,d_rev_residual_capacity,d_parallel_edge,d_residual_capacity,d_excess,d_height,d_set2_ptr,d_set2,d_partB);
  }

  cudaDeviceSynchronize();
  pp_bfs_init<<<numBlocks, numThreads>>>(source,sink,V,d_excess, d_partB,d_level,d_height);
  int hops = 0;
  do{
    bfsflag = false;
    cudaMemcpyToSymbol(::bfsflag, &bfsflag, sizeof(bool), 0, cudaMemcpyHostToDevice);
    pp_bfs<<<numBlocks, numThreads>>>(source,sink,hops,V,d_meta,d_data,d_level,d_residual_capacity,d_rev_residual_capacity,d_height,d_partB);
    cudaDeviceSynchronize();
    hops++;
    cudaMemcpyFromSymbol(&bfsflag, ::bfsflag, sizeof(bool), 0, cudaMemcpyDeviceToHost);
  }while(bfsflag);

cudaDeviceSynchronize();
    cudaMemset(d_set_ptr,0,sizeof(int));
    cudaMemset(d_set2_ptr,0,sizeof(int));
create_pp_wl<<<numBlocks, numThreads>>>(source,sink,d_excess,d_set_ptr,d_set2_ptr,d_set,d_set2,d_height, V, d_partB);
cudaDeviceSynchronize();
    active = 0;
    cudaMemcpy(&active,d_set_ptr,sizeof( int),cudaMemcpyDeviceToHost);
    active2 = 0;
    cudaMemcpy(&active2,d_set2_ptr,sizeof( int),cudaMemcpyDeviceToHost);
}
cudaDeviceSynchronize();
cudaStreamDestroy(push);
cudaStreamDestroy(pull);

}


void part_final(int source, int sink,int *d_level, int *d_height, int *d_meta, int *d_data, int *d_residual_capacity, 
int *d_rev_residual_capacity, bool *d_partB, int *d_excess, int V, 
int *d_set, int *d_set_ptr,int *d_part, int *d_part_ptr, int kernel_cycles, int *d_parallel_edge, bool *d_partfinal){
  unsigned numThreads   = (V< THREADS_PER_BLOCK)?V: THREADS_PER_BLOCK;
  unsigned numBlocks    = (V+numThreads-1)/numThreads;

  find_part<<<numBlocks, numThreads>>>(V, d_level,d_height, d_partB, d_part, d_part_ptr,d_partfinal);
  int nodes = 0;
  cudaDeviceSynchronize();
  cudaMemcpy(&nodes,d_part_ptr,sizeof(int),cudaMemcpyDeviceToHost);
  if(nodes==0) return;

  unsigned numPartThreads   = (nodes< THREADS_PER_BLOCK)?nodes: THREADS_PER_BLOCK;
  unsigned numPartBlocks    = (nodes+numThreads-1)/numThreads;

  bool bfsflag = false;
  final_bfs_init<<<numPartBlocks, numPartThreads>>>(d_part,nodes,source, sink,V,d_excess, d_partB,d_level,d_height);
  int hops = 0;
  do{
    bfsflag = false;
    cudaMemcpyToSymbol(::bfsflag, &bfsflag, sizeof(bool), 0, cudaMemcpyHostToDevice);
    final_bfs<<<numPartBlocks, numPartThreads>>>(d_part, nodes,source,sink,hops,V,d_meta,d_data,d_level,d_residual_capacity,d_rev_residual_capacity,d_height,d_partB);
    cudaDeviceSynchronize();
    hops++;
    cudaMemcpyFromSymbol(&bfsflag, ::bfsflag, sizeof(bool), 0, cudaMemcpyDeviceToHost);
  }while(bfsflag);
    cudaMemset(d_set_ptr,0,sizeof(int));
    create_final_wl<<<numPartBlocks, numPartThreads>>>(d_part,nodes,source, sink,d_excess,V,d_set_ptr,d_height,d_set);
    cudaDeviceSynchronize();
    int active = 0;
    cudaMemcpy(&active,d_set_ptr,sizeof( int),cudaMemcpyDeviceToHost);
      
  int cnt = 0;
  while(active>0){
    cnt++;
    unsigned activenumThreads   = (active < THREADS_PER_BLOCK)? active: THREADS_PER_BLOCK;
    unsigned activenumBlocks    = (active+activenumThreads-1)/activenumThreads;
    push_kernel<<<activenumBlocks,activenumThreads>>>(kernel_cycles,V,d_meta,d_data,d_rev_residual_capacity,d_excess,d_parallel_edge,d_residual_capacity,d_height,d_set_ptr,d_set);
    remove_push_final<<<activenumBlocks,activenumThreads>>>(d_meta,d_data,d_rev_residual_capacity,d_parallel_edge,d_residual_capacity,d_excess,d_height,d_set_ptr,d_set,d_partfinal);
    cudaDeviceSynchronize();

    final_bfs_init<<<numPartBlocks, numPartThreads>>>(d_part,nodes,source, sink,V,d_excess, d_partB,d_level,d_height);
    cudaDeviceSynchronize();
    int hops = 0;
    do{
      bfsflag = false;
      cudaMemcpyToSymbol(::bfsflag, &bfsflag, sizeof(bool), 0, cudaMemcpyHostToDevice);
      final_bfs<<<numPartBlocks, numPartThreads>>>(d_part, nodes,source,sink,hops,V,d_meta,d_data,d_level,d_residual_capacity,d_rev_residual_capacity,d_height,d_partB);
      cudaDeviceSynchronize();
      hops++;
      cudaMemcpyFromSymbol(&bfsflag, ::bfsflag, sizeof(bool), 0, cudaMemcpyDeviceToHost);
    }while(bfsflag);
    cudaMemset(d_set_ptr,0,sizeof(int));
    create_final_wl<<<numPartBlocks, numPartThreads>>>(d_part,nodes,source, sink,d_excess,V,d_set_ptr,d_height,d_set);
    cudaDeviceSynchronize();
    active = 0;
    cudaMemcpy(&active,d_set_ptr,sizeof( int),cudaMemcpyDeviceToHost);
  }

  find_final_cut<<<numBlocks, numThreads>>>( V,d_level ,d_partB);
  cudaDeviceSynchronize();
}


void staticMaxFlow(graph& g,int source0,int sink0,int kernel_cycles0,
int *d_meta, int *d_data, int *d_residual_capacity, int *d_rev_residual_capacity,
int *d_parallel_edge, int *d_excess, int *d_level, int *d_height, int *d_weight, 
int *d_set, int *d_set2,int *d_set_ptr, int *d_set2_ptr, bool *d_partB)
{
  // CSR BEGIN
  int V = g.num_nodes();
  int E = g.num_edges();



  unsigned numThreads   = (V < THREADS_PER_BLOCK)? V: THREADS_PER_BLOCK;
  unsigned numBlocks    = (V+numThreads-1)/numThreads;


    cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  cudaEventRecord(start,0);

  init_kernel<<<numBlocks,numThreads>>>(V,d_meta, d_data,d_weight,d_residual_capacity,d_rev_residual_capacity,d_parallel_edge, d_excess);
  staticMaxFlow_kernel_7<<<1,1>>>(source0, sink0,d_meta,d_data,d_residual_capacity,d_excess,d_parallel_edge,d_rev_residual_capacity);

  cudaDeviceSynchronize();
  
  my_init_1<<<numBlocks,numThreads>>>(V,d_height,d_level,d_excess,source0,sink0);
  cudaDeviceSynchronize();
  int hops = 0; 
  bool bfsflag = false;
  do{
    bfsflag = false;
    cudaMemcpyToSymbol(::bfsflag, &bfsflag, sizeof(bool), 0, cudaMemcpyHostToDevice);
    staticMaxFlow_kernel_11<<<numBlocks, numThreads>>>(hops,V,d_meta,d_data,d_level,d_rev_residual_capacity,d_height);
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&bfsflag, ::bfsflag, sizeof(bool), 0, cudaMemcpyDeviceToHost);
    hops++;
  }while(bfsflag);
  cudaMemset(d_set_ptr,0,sizeof(int));
  create_wl<<<numBlocks, numThreads>>>(source0, sink0,d_excess,V,d_set_ptr,d_height,d_set);
  cudaDeviceSynchronize();
  int active = 0;
  cudaMemcpy(&active,d_set_ptr,sizeof( int),cudaMemcpyDeviceToHost);

  int cnt = 0;
  while(active>0){

        int activeThreadsPerBlock = (active < THREADS_PER_BLOCK)? active: THREADS_PER_BLOCK;
    unsigned activeBlocks    = (active+activeThreadsPerBlock-1)/activeThreadsPerBlock;

    push_kernel<<<activeBlocks, activeThreadsPerBlock>>>(kernel_cycles0,V,d_meta,d_data,d_rev_residual_capacity,d_excess,d_parallel_edge,d_residual_capacity,d_height,d_set_ptr,d_set);
    remove_push<<<activeBlocks,activeThreadsPerBlock>>>(d_meta,d_data,d_rev_residual_capacity,d_parallel_edge,d_residual_capacity,d_excess,d_height,d_set_ptr,d_set);
    cudaDeviceSynchronize();
    my_init_1<<<numBlocks,numThreads>>>(V,d_height,d_level,d_excess,source0,sink0);
    cudaDeviceSynchronize();
    int hops = 0; 

    bool bfsflag = false;

    do{
      bfsflag = false;
      cudaMemcpyToSymbol(::bfsflag, &bfsflag, sizeof(bool), 0, cudaMemcpyHostToDevice);
      staticMaxFlow_kernel_11<<<numBlocks, numThreads>>>(hops,V,d_meta,d_data,d_level,d_rev_residual_capacity,d_height);
      cudaDeviceSynchronize();
      cudaMemcpyFromSymbol(&bfsflag, ::bfsflag, sizeof(bool), 0, cudaMemcpyDeviceToHost);
      hops++;

    }while(bfsflag);

    cudaMemset(d_set_ptr,0,sizeof(int));
    create_wl<<<numBlocks, numThreads>>>(source0, sink0,d_excess,V,d_set_ptr,d_height,d_set);
    cudaDeviceSynchronize();
    active = 0;
    cudaMemcpy(&active,d_set_ptr,sizeof( int),cudaMemcpyDeviceToHost);
    cnt++;
  }
  find_cut<<<numBlocks, numThreads>>>( V,d_level ,d_partB);
  cudaDeviceSynchronize();

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("static main push cnt:%d GPU Time: %.6f ms\n",cnt ,milliseconds);

} 


void Incremental(int V,int *d_meta, int *d_data, int *d_parallel_edge, int *d_residual_capacity, int *d_rev_residual_capacity, int *d_weight, int *d_excess,
bool *d_partB,int *d_level, int *d_height,int *d_set, int *d_set_ptr, int *d_set2, int *d_set2_ptr, int source, int sink, int kernel_cycles, bool *d_partfinal)
{

  unsigned numThreads   = (V < THREADS_PER_BLOCK)? V: THREADS_PER_BLOCK;
  unsigned numBlocks    = (V+numThreads-1)/numThreads;

  incremental_kernel_20<<<numBlocks, numThreads>>>(V,  d_meta, d_data,d_rev_residual_capacity, d_residual_capacity, d_partB, d_excess,d_weight);

part_pp(source, sink,d_level, d_height,d_meta,d_data, d_residual_capacity, d_rev_residual_capacity, d_partB, d_excess,  V, 
d_set, d_set_ptr,d_set2, d_set2_ptr,kernel_cycles,d_parallel_edge);
part_final(source, sink,d_level,d_height,d_meta,d_data,d_residual_capacity, 
d_rev_residual_capacity, d_partB,d_excess, V, 
d_set, d_set_ptr,d_set2,d_set2_ptr,kernel_cycles,d_parallel_edge, d_partfinal);


}

__global__ void update_kernel(update* d_updateBatch, int batchelements, int* d_meta, int* d_data, int* d_weight){ // BEGIN KER FUN via ADDKERNEL
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= batchelements) return;
  update u = d_updateBatch[tid];
  if( u.type!='a') return;
  int uedge = -1; // DEVICE ASSTMENT in .h

  int src = u.source; // DEVICE ASSTMENT in .h

  int dest = u.destination; // DEVICE ASSTMENT in .h

  int new_capacity = u.weight; // DEVICE ASSTMENT in .h

  for (int edge = d_meta[src]; edge < d_meta[src+1]; edge++) { // FOR NBR ITR 
    int dd = d_data[edge];
    if (dd == dest){ // if filter begin 
      uedge = edge;

    } // if filter end

  } //  end FOR NBR ITR. TMP FIX
  d_weight[uedge] = new_capacity;
} // end KER FUNC

int static_recalculate(graph& g, std::vector<update> updateBatch, int batchSize, int source0, 
  int sink0, int kernel_cycles0)
{
   int V = g.num_nodes();
   int E = g.num_edges();

  int *h_meta = g.indexofNodes;
  int *h_data = g.edgeList;
  int *h_weight = g.getEdgeLen();
  int *h_parallel_edge = g.parallel_edge;

  int* d_parallel_edge;
  int *d_meta;
  int *d_data;
  int *d_weight;
  cudaMalloc(&d_meta,sizeof(int)*(V+1));
  cudaMalloc(&d_data,sizeof(int)*(E));
  cudaMalloc(&d_weight,sizeof(int)*(E));
  cudaMalloc(&d_parallel_edge, sizeof(int)*(E));


  cudaMemcpy(d_parallel_edge, h_parallel_edge, sizeof(int)*(E), cudaMemcpyHostToDevice);
  cudaMemcpy(d_meta,h_meta,sizeof(int)*(V+1),cudaMemcpyHostToDevice);
  cudaMemcpy(d_data,h_data,sizeof(int)*(E),cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight,h_weight,sizeof(int)*(E),cudaMemcpyHostToDevice);


    int* d_residual_capacity;
  cudaMalloc(&d_residual_capacity, sizeof(int)*(E));

  int* d_rev_residual_capacity;
  cudaMalloc(&d_rev_residual_capacity, sizeof(int)*(E));

  int* d_excess;
  cudaMalloc(&d_excess, sizeof(int)*(V));


  int* d_height;
  cudaMalloc(&d_height, sizeof(int)*(V));

  int* d_level;
  cudaMalloc(&d_level, sizeof(int)*(V));



      int *d_set_ptr;
  cudaMalloc(&d_set_ptr,sizeof(int));
  
  int *d_set;
  cudaMalloc(&d_set,sizeof(int)*V);






  unsigned numThreads   = (V < THREADS_PER_BLOCK)? V: THREADS_PER_BLOCK;
  unsigned numBlocks    = (V+numThreads-1)/numThreads;


    // TIMER START

  int _batchSize = batchSize;
  update *d_updateBatch;
  cudaMalloc(&d_updateBatch,sizeof(update)*_batchSize);
  int batchElements = 0;


  int true_cnt = 0;
  int cnt = 0;
  for( int updateIndex = 0 ; updateIndex < updateBatch.size() ; updateIndex += _batchSize){
    if((updateIndex + _batchSize) > updateBatch.size())
    {
      batchElements = updateBatch.size() - updateIndex ;
      
    }
    else
    batchElements = _batchSize ;
    cudaMemcpy(d_updateBatch,&updateBatch[updateIndex],batchElements*sizeof(update),cudaMemcpyHostToDevice);
    unsigned updateThreads = (batchElements < THREADS_PER_BLOCK)? batchElements: THREADS_PER_BLOCK;
    unsigned updateBlocks = (batchElements+updateThreads-1)/updateThreads;
    if(batchElements<_batchSize) true_cnt--; 

    update_kernel<<<updateBlocks,updateThreads>>>(d_updateBatch,batchElements, d_meta, d_data, d_weight);
    cudaDeviceSynchronize();

  }


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  cudaEventRecord(start,0);


  init_kernel<<<numBlocks,numThreads>>>(V,d_meta, d_data,d_weight,d_residual_capacity,d_rev_residual_capacity,d_parallel_edge, d_excess);
  staticMaxFlow_kernel_7<<<1,1>>>(source0, sink0,d_meta,d_data,d_residual_capacity,d_excess,d_parallel_edge,d_rev_residual_capacity);

  cudaDeviceSynchronize();

  my_init_1<<<numBlocks,numThreads>>>(V,d_height,d_level,d_excess,source0,sink0);
  cudaDeviceSynchronize();
  int hops = 0; 
  bool bfsflag = false;
  do{
    bfsflag = false;
    cudaMemcpyToSymbol(::bfsflag, &bfsflag, sizeof(bool), 0, cudaMemcpyHostToDevice);
    staticMaxFlow_kernel_11<<<numBlocks, numThreads>>>(hops,V,d_meta,d_data,d_level,d_rev_residual_capacity,d_height);
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&bfsflag, ::bfsflag, sizeof(bool), 0, cudaMemcpyDeviceToHost);
    hops++;
  }while(bfsflag);
  cudaMemset(d_set_ptr,0,sizeof(int));
  create_wl<<<numBlocks, numThreads>>>(source0, sink0,d_excess,V,d_set_ptr,d_height,d_set);
  cudaDeviceSynchronize();
  int active = 0;
  cudaMemcpy(&active,d_set_ptr,sizeof( int),cudaMemcpyDeviceToHost);


  while(active>0){

        int activeThreadsPerBlock = (active < THREADS_PER_BLOCK)? active: THREADS_PER_BLOCK;
    unsigned activeBlocks    = (active+activeThreadsPerBlock-1)/activeThreadsPerBlock;

    push_kernel<<<activeBlocks, activeThreadsPerBlock>>>(kernel_cycles0,V,d_meta,d_data,d_rev_residual_capacity,d_excess,d_parallel_edge,d_residual_capacity,d_height,d_set_ptr,d_set);
    remove_push<<<activeBlocks,activeThreadsPerBlock>>>(d_meta,d_data,d_rev_residual_capacity,d_parallel_edge,d_residual_capacity,d_excess,d_height,d_set_ptr,d_set);
    cudaDeviceSynchronize();
    my_init_1<<<numBlocks,numThreads>>>(V,d_height,d_level,d_excess,source0,sink0);
    cudaDeviceSynchronize();
    int hops = 0; 

    bool bfsflag = false;

    do{
      bfsflag = false;
      cudaMemcpyToSymbol(::bfsflag, &bfsflag, sizeof(bool), 0, cudaMemcpyHostToDevice);
      staticMaxFlow_kernel_11<<<numBlocks, numThreads>>>(hops,V,d_meta,d_data,d_level,d_rev_residual_capacity,d_height);
      cudaDeviceSynchronize();
      cudaMemcpyFromSymbol(&bfsflag, ::bfsflag, sizeof(bool), 0, cudaMemcpyDeviceToHost);
      hops++;

    }while(bfsflag);

    cudaMemset(d_set_ptr,0,sizeof(int));
    create_wl<<<numBlocks, numThreads>>>(source0, sink0,d_excess,V,d_set_ptr,d_height,d_set);
    cudaDeviceSynchronize();
    active = 0;
    cudaMemcpy(&active,d_set_ptr,sizeof( int),cudaMemcpyDeviceToHost);
    cnt++;
  }




  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("static  UPDATE GPU Time: %.6f ms\n", milliseconds);
    recalculate_max_flow_kernel_39<<<numBlocks, numThreads>>>(source0,sink0,V,E,d_excess);
  cudaDeviceSynchronize();


    int flow;
  cudaMemcpy(&flow,&d_excess[sink0],sizeof(int),cudaMemcpyDeviceToHost);
   cudaFree(d_meta);
  cudaFree(d_data);
  cudaFree(d_weight);

  cudaFree(d_residual_capacity);
  cudaFree(d_rev_residual_capacity);
  cudaFree(d_excess);
  cudaFree(d_parallel_edge);
  cudaFree(d_height);
  cudaFree(d_level);
  cudaFree(d_updateBatch);
  cudaFree(d_set);
  cudaFree(d_set_ptr);
  
  return flow;
} 




int recalculate_max_flow(graph& g, std::vector<update> updateBatch, int batchSize, int source, 
  int sink, int kernel_cycles)
{
   int V = g.num_nodes();
   int E = g.num_edges();

  int *h_meta = g.indexofNodes;
  int *h_data = g.edgeList;
  int *h_weight = g.getEdgeLen();
  int *h_parallel_edge = g.parallel_edge;

  int* d_parallel_edge;
  int *d_meta;
  int *d_data;
  int *d_weight;
  cudaMalloc(&d_meta,sizeof(int)*(V+1));
  cudaMalloc(&d_data,sizeof(int)*(E));
  cudaMalloc(&d_weight,sizeof(int)*(E));
  cudaMalloc(&d_parallel_edge, sizeof(int)*(E));


  cudaMemcpy(d_parallel_edge, h_parallel_edge, sizeof(int)*(E), cudaMemcpyHostToDevice);
  cudaMemcpy(d_meta,h_meta,sizeof(int)*(V+1),cudaMemcpyHostToDevice);
  cudaMemcpy(d_data,h_data,sizeof(int)*(E),cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight,h_weight,sizeof(int)*(E),cudaMemcpyHostToDevice);
  unsigned numThreads   = (V < THREADS_PER_BLOCK)? V: THREADS_PER_BLOCK;
  unsigned numBlocks    = (V+numThreads-1)/numThreads;
  int* d_residual_capacity;
  cudaMalloc(&d_residual_capacity, sizeof(int)*(E));

  int* d_rev_residual_capacity;
  cudaMalloc(&d_rev_residual_capacity, sizeof(int)*(E));

  int* d_excess;
  cudaMalloc(&d_excess, sizeof(int)*(V));


  int* d_height;
  cudaMalloc(&d_height, sizeof(int)*(V));

  int* d_level;
  cudaMalloc(&d_level, sizeof(int)*(V));


      int *d_set_ptr;
  cudaMalloc(&d_set_ptr,sizeof(int));
  
  int *d_set2;
  cudaMalloc(&d_set2,sizeof(int)*V);
        int *d_set2_ptr;
  cudaMalloc(&d_set2_ptr,sizeof(int));
  
  int *d_set;
  cudaMalloc(&d_set,sizeof(int)*V);



  bool *d_partB;
  cudaMalloc(&d_partB,sizeof(bool)*V);
    bool *d_partfinal;
  cudaMalloc(&d_partfinal,sizeof(bool)*V);


  staticMaxFlow(g,source,sink,kernel_cycles,
d_meta,d_data,d_residual_capacity,d_rev_residual_capacity,
d_parallel_edge, d_excess, d_level, d_height,d_weight, 
d_set, d_set2,d_set_ptr,d_set2_ptr,d_partB);

  int _batchSize = batchSize;
  update *d_updateBatch;
  cudaMalloc(&d_updateBatch,sizeof(update)*_batchSize);
  int batchElements = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  cudaEventRecord(start,0);
  float total = 0;
  int true_cnt = 0;
  int cnt = 0;
  for( int updateIndex = 0 ; updateIndex < updateBatch.size() ; updateIndex += _batchSize){
    if((updateIndex + _batchSize) > updateBatch.size())
    {
      batchElements = updateBatch.size() - updateIndex ;
      
    }
    else
    batchElements = _batchSize ;
    cudaMemcpy(d_updateBatch,&updateBatch[updateIndex],batchElements*sizeof(update),cudaMemcpyHostToDevice);
    unsigned updateThreads = (batchElements < THREADS_PER_BLOCK)? batchElements: THREADS_PER_BLOCK;
    unsigned updateBlocks = (batchElements+updateThreads-1)/updateThreads;
    if(batchElements<_batchSize) true_cnt--; 

  cudaEvent_t start1, stop1;
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);
  float milliseconds1 = 0;
  cudaEventRecord(start1,0);

    OnAdd_kernel<<<updateBlocks,updateThreads>>>(d_updateBatch,batchElements, d_meta, d_data, d_weight, d_rev_residual_capacity, d_parallel_edge, d_residual_capacity);
    cudaDeviceSynchronize();

    cudaDeviceSynchronize();
    Incremental(V,d_meta,d_data,d_parallel_edge,d_residual_capacity,d_rev_residual_capacity, d_weight,d_excess,
d_partB,d_level,d_height,d_set,d_set_ptr, d_set2,d_set2_ptr, source, sink,kernel_cycles,d_partfinal);
  cudaEventRecord(stop1,0);
  cudaEventSynchronize(stop1);
  cudaEventElapsedTime(&milliseconds1, start1, stop1);
total+=milliseconds1;
  true_cnt++;
  cnt++;

  }
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("total GPU Time: %.6f ms\n", milliseconds);

  printf("DYN TIME:%.6fms true_cnt=%d cnt=%d per batch time:%.6f\n",total,true_cnt,cnt,total/cnt);

  recalculate_max_flow_kernel_39<<<numBlocks, numThreads>>>(source,sink,V,E,d_excess);
  cudaDeviceSynchronize();


    int flow;
  cudaMemcpy(&flow,&d_excess[sink],sizeof(int),cudaMemcpyDeviceToHost);
  printf("Dynamic flow:%d\n",flow);

 
  cudaFree(d_meta);
  cudaFree(d_data);
  cudaFree(d_weight);

  cudaFree(d_residual_capacity);
  cudaFree(d_rev_residual_capacity);
  cudaFree(d_excess);
  cudaFree(d_parallel_edge);
  cudaFree(d_height);
  cudaFree(d_level);

  cudaFree(d_updateBatch);
  cudaFree(d_set);
  cudaFree(d_set_ptr);

  cudaFree(d_set2);
  cudaFree(d_set2_ptr);
  cudaFree(d_partB);
  cudaFree(d_partfinal);
   return flow;



}


int  main( int  argc, char** argv) {
  char* originalgraph=argv[1];
  char update_filename[MAX_FILENAME_LEN];
char* prefix = argv[2];  // updatefile prefix from command line

  int source = atoi(argv[3]);
  int sink = atoi(argv[4]);
  printf("source:%d sink:%d\n",source,sink);

  graph G1(originalgraph);
    G1.loadGraph();
      int V = G1.num_nodes();
  int E = G1.num_edges();

      for( int i = 1;i<11;i++){
memset(update_filename, 0, sizeof(update_filename));

    strcpy(update_filename, prefix);
    char suffix[32];
    sprintf(suffix, "_p%d.bin", i);
    strcat(update_filename, suffix);

    printf("Processing file: %s\n", update_filename);
      int dynflow;
      int statflow;
      

      std::vector<update> updateEdges=G1.parseUpdates(update_filename);

      int updatesSize = updateEdges.size();
      printf("updates size:%d\n",updatesSize);
  int avg_deg = G1.num_edges()/G1.num_nodes();

if(avg_deg>1){
      printf("=======kernel cycles %d =======\n",avg_deg-1);
  

        dynflow= recalculate_max_flow(G1,updateEdges,updatesSize,source,sink,avg_deg-1);
        statflow = static_recalculate(G1,updateEdges,updatesSize,source,sink,avg_deg-1);
        if(dynflow!=statflow){
          printf("<<<<<<<<<<<<<<<<<<<<<<<<PANIC NOT MATCHING dynflow:%d statflow:%d>>>>>>>>>>>>>>>>>>>>>>>>>>>\n",dynflow,statflow);
        }
  }

  if(avg_deg>0){
      printf("=======kernel cycles %d =======\n",avg_deg);
  

        dynflow= recalculate_max_flow(G1,updateEdges,updatesSize,source,sink,avg_deg);
        statflow = static_recalculate(G1,updateEdges,updatesSize,source,sink,avg_deg);
        if(dynflow!=statflow){
          printf("<<<<<<<<<<<<<<<<<<<<<<<<PANIC NOT MATCHING dynflow:%d statflow:%d>>>>>>>>>>>>>>>>>>>>>>>>>>>\n",dynflow,statflow);
        }

  }


      printf("=======kernel cycles %d =======\n",avg_deg+1);
  

        dynflow= recalculate_max_flow(G1,updateEdges,updatesSize,source,sink,avg_deg+1);
        statflow = static_recalculate(G1,updateEdges,updatesSize,source,sink,avg_deg+1);
        if(dynflow!=statflow){
          printf("<<<<<<<<<<<<<<<<<<<<<<<<PANIC NOT MATCHING dynflow:%d statflow:%d>>>>>>>>>>>>>>>>>>>>>>>>>>>\n",dynflow,statflow);
        }
    
       
    }


  return 0;
}