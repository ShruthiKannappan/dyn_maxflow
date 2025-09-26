// FOR BC: nvcc bc_dsl_v2.cu -arch=sm_60 -std=c++14 -rdc=true # HW must support CC 6.0+ Pascal or after
#include "dyn_data.h"


#define THREADS_PER_BLOCK 1024

const int MAX_FILENAME_LEN = 1024;
void staticMaxFlow(int V, int E,int source0,int sink0,int* d_residual_capacity,
  int* d_rev_residual_capacity,int* d_excess,
  int kernel_cycles0, int *d_meta, int *d_data, int *d_weight,int *d_parallel_edge, int *d_height, int *d_level, int *d_set_ptr, int *d_set)

{


  // CSR END
  //LAUNCH CONFIG
  unsigned numThreads   = (V < THREADS_PER_BLOCK)? V: THREADS_PER_BLOCK;
  unsigned numBlocks    = (V+numThreads-1)/numThreads;


    // TIMER START
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  cudaEventRecord(start,0);

 staticMaxFlow_kernel_1<<<numBlocks, numThreads>>>(V,d_meta,d_data,d_weight,d_residual_capacity,d_rev_residual_capacity,d_parallel_edge,d_excess);

  staticMaxFlow_kernel_7<<<1,1>>>(source0,V,E,d_meta,d_data,d_residual_capacity,d_excess,d_parallel_edge,d_rev_residual_capacity);


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

    staticMaxFlow_kernel_14<<<activeBlocks, activeThreadsPerBlock>>>(source0,sink0,kernel_cycles0,V,E,d_meta,d_data,d_rev_residual_capacity,d_excess,d_parallel_edge,d_residual_capacity,d_height,d_set_ptr,d_set);
    staticMaxFlow_kernel_17<<<activeBlocks, activeThreadsPerBlock>>>(source0,sink0,V,E,d_meta,d_data,d_rev_residual_capacity,d_parallel_edge,d_residual_capacity,d_excess,d_height,d_set_ptr,d_set);
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

  //TIMER STOP
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("static GPU Time: %.6f ms\n", milliseconds);



} //end FUN
void Incremental( int V,int E,int* d_meta,int* d_data,int* d_weight, int source1, int sink1, int* d_parallel_edge, 
  int* d_residual_capacity, int* d_rev_residual_capacity, int* d_excess, 
  int* d_height, int* d_level, int kernel_cycles1, int *d_set,int *d_set_ptr)
{

  unsigned numThreads   = (V < THREADS_PER_BLOCK)? V: THREADS_PER_BLOCK;
  unsigned numBlocks    = (V+numThreads-1)/numThreads;


  incremental_kernel_20<<<numBlocks, numThreads>>>(V,E,d_meta,d_data,d_rev_residual_capacity,d_residual_capacity);
  incremental_kernel_22<<<numBlocks, numThreads>>>(V,E,d_meta,d_data,d_weight,d_residual_capacity,d_excess,d_parallel_edge);
  staticMaxFlow_kernel_7<<<1,1>>>(source1,V,E,d_meta,d_data,d_residual_capacity,d_excess,d_parallel_edge,d_rev_residual_capacity);
  cudaDeviceSynchronize();


  my_init_1<<<numBlocks,numThreads>>>(V,d_height,d_level,d_excess,source1,sink1);
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
    create_wl<<<numBlocks, numThreads>>>(source1, sink1,d_excess,V,d_set_ptr,d_height,d_set);
    cudaDeviceSynchronize();
    int active = 0;
    cudaMemcpy(&active,d_set_ptr,sizeof( int),cudaMemcpyDeviceToHost);

  int cnt = 0;
  while(active>0){

        int activeThreadsPerBlock = (active < THREADS_PER_BLOCK)? active: THREADS_PER_BLOCK;
    unsigned activeBlocks    = (active+activeThreadsPerBlock-1)/activeThreadsPerBlock;

    staticMaxFlow_kernel_14<<<activeBlocks, activeThreadsPerBlock>>>(source1,sink1,kernel_cycles1,V,E,d_meta,d_data,d_rev_residual_capacity,d_excess,d_parallel_edge,d_residual_capacity,d_height,d_set_ptr,d_set);
    staticMaxFlow_kernel_17<<<activeBlocks, activeThreadsPerBlock>>>(source1,sink1,V,E,d_meta,d_data,d_rev_residual_capacity,d_parallel_edge,d_residual_capacity,d_excess,d_height,d_set_ptr,d_set);
    cudaDeviceSynchronize();


    my_init_1<<<numBlocks,numThreads>>>(V,d_height,d_level,d_excess,source1,sink1);
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
    create_wl<<<numBlocks, numThreads>>>(source1, sink1,d_excess,V,d_set_ptr,d_height,d_set);
    cudaDeviceSynchronize();
    active = 0;
    cudaMemcpy(&active,d_set_ptr,sizeof( int),cudaMemcpyDeviceToHost);
    cnt++;
  }


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
  
  int *d_set;
  cudaMalloc(&d_set,sizeof(int)*V);

  staticMaxFlow(V,E,source,sink,d_residual_capacity,d_rev_residual_capacity,d_excess,kernel_cycles,d_meta,d_data,d_weight,d_parallel_edge,d_height,d_level,d_set_ptr,d_set);
  cudaDeviceSynchronize();
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
    Incremental(V,E,d_meta,d_data,d_weight,source,sink,d_parallel_edge,d_residual_capacity,d_rev_residual_capacity,d_excess,d_height,d_level,kernel_cycles,d_set,d_set_ptr);
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


   return flow;



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
  int sink0, int kernel_cycles)
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
  
  int *d_set;
  cudaMalloc(&d_set,sizeof(int)*V);

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


staticMaxFlow_kernel_1<<<numBlocks, numThreads>>>(V,d_meta,d_data,d_weight,d_residual_capacity,d_rev_residual_capacity,d_parallel_edge,d_excess);

  staticMaxFlow_kernel_7<<<1,1>>>(source0,V,E,d_meta,d_data,d_residual_capacity,d_excess,d_parallel_edge,d_rev_residual_capacity);

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

    staticMaxFlow_kernel_14<<<activeBlocks, activeThreadsPerBlock>>>(source0,sink0,kernel_cycles,V,E,d_meta,d_data,d_rev_residual_capacity,d_excess,d_parallel_edge,d_residual_capacity,d_height,d_set_ptr,d_set);
    staticMaxFlow_kernel_17<<<activeBlocks, activeThreadsPerBlock>>>(source0,sink0,V,E,d_meta,d_data,d_rev_residual_capacity,d_parallel_edge,d_residual_capacity,d_excess,d_height,d_set_ptr,d_set);
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
  printf("STATIC UPDATE GPU Time: %.6f ms\n", milliseconds);

    int flow;
  cudaMemcpy(&flow,&d_excess[sink0],sizeof(int),cudaMemcpyDeviceToHost);
  // printf("Dynamic flow:%d\n",flow);


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


int  main( int  argc, char** argv) {
  char* originalgraph=argv[1];
  char update_filename[MAX_FILENAME_LEN];
char* prefix = argv[2];  // updatefile prefix from command line

  int source = atoi(argv[3]);
  int sink = atoi(argv[4]);
  printf("source:%d sink:%d\n",source,sink);
  // return 0;
  graph G1(originalgraph);
    G1.loadGraph();
      int V = G1.num_nodes();
  int E = G1.num_edges();

    // return 0;

    for( int i = 1;i<11;i++){
memset(update_filename, 0, sizeof(update_filename));

    // Copy prefix and append "_p{i}.txt"
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