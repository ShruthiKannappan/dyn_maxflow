// FOR BC: nvcc bc_dsl_v2.cu -arch=sm_60 -std=c++14 -rdc=true # HW must support CC 6.0+ Pascal or after
#include "dyn_topo.h"

const int MAX_FILENAME_LEN = 1024;
#define THREADS_PER_BLOCK 1024
void checkCudaError( int  i)
{       
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)  
  {   
    printf("%d , CUDA error: %s\n", i, cudaGetErrorString(error));
    exit(0);
  } 
} 

void staticMaxFlow(int V, int E,int source0,int sink0,int* d_residual_capacity,
  int* d_rev_residual_capacity,int* d_excess,
  int kernel_cycles0, int *d_meta, int *d_data, int *d_weight, int *d_parallel_edge, int *d_level, int *d_height)

{



  //LAUNCH CONFIG
  unsigned numThreads   = (V < THREADS_PER_BLOCK)? V: THREADS_PER_BLOCK;
  unsigned numBlocks    = (V+numThreads-1)/numThreads;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  cudaEventRecord(start,0);

 staticMaxFlow_kernel_1<<<numBlocks, numThreads>>>(V,d_meta,d_data,d_weight,d_residual_capacity,d_rev_residual_capacity,d_parallel_edge,d_excess);

  staticMaxFlow_kernel_7<<<1,1>>>(source0,V,E,d_meta,d_data,d_residual_capacity,d_excess,d_parallel_edge,d_rev_residual_capacity);


  bool flag1 = true; // asst in .cu
  cudaDeviceSynchronize();
  do{
    flag1 = false;
    int hops = 0; // asst in .cu
    staticMaxFlow_kernel_10<<<numBlocks, numThreads>>>(source0,sink0,V,E,d_excess,d_level,d_height);
    cudaDeviceSynchronize();
    bool bfsflag = false; // asst in .cu
    do{
      bfsflag = false;
      cudaMemcpyToSymbol(::bfsflag, &bfsflag, sizeof(bool), 0, cudaMemcpyHostToDevice);
      staticMaxFlow_kernel_11<<<numBlocks, numThreads>>>(hops,V,E,d_meta,d_data,d_residual_capacity,d_level,d_rev_residual_capacity,d_height);
      cudaDeviceSynchronize();
      cudaMemcpyFromSymbol(&bfsflag, ::bfsflag, sizeof(bool), 0, cudaMemcpyDeviceToHost);
      hops++;
    }while(bfsflag);

    cudaMemcpyToSymbol(::flag1, &flag1, sizeof(bool), 0, cudaMemcpyHostToDevice);
    staticMaxFlow_kernel_14<<<numBlocks, numThreads>>>(source0,sink0,kernel_cycles0,V,E,d_meta,d_data,d_excess,d_parallel_edge,d_rev_residual_capacity,d_residual_capacity,d_height);
      staticMaxFlow_kernel_17<<<numBlocks, numThreads>>>(source0,sink0,V,E,d_meta,d_data,d_rev_residual_capacity,d_parallel_edge,d_residual_capacity,d_excess,d_height);
     cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&flag1, ::flag1, sizeof(bool), 0, cudaMemcpyDeviceToHost);
  }while(flag1);


  //TIMER STOP
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("static GPU Time: %.6f ms\n", milliseconds);


          int flow;
  cudaMemcpy(&flow,&d_excess[sink0],sizeof(int),cudaMemcpyDeviceToHost);
  printf("flow:%d  kernel cycles:%d\n",flow,kernel_cycles0);

} //end FUN
void Incremental(graph& g, int V,int E,int* d_meta,int* d_data,int* d_weight, int source1, int sink1, int* d_parallel_edge, 
  int* d_residual_capacity, int* d_rev_residual_capacity, int* d_excess, 
  int* d_height, int* d_level, int kernel_cycles1)
{

  unsigned numThreads   = (V < THREADS_PER_BLOCK)? V: THREADS_PER_BLOCK;
  unsigned numBlocks    = (V+numThreads-1)/numThreads;

  int num_nodes1 = g.num_nodes(); // asst in .cu
  
  incremental_kernel_20<<<numBlocks, numThreads>>>(V,E,d_meta,d_data,d_rev_residual_capacity,d_residual_capacity);
  incremental_kernel_22<<<numBlocks, numThreads>>>(V,E,d_meta,d_weight,d_residual_capacity,d_excess);
  
  staticMaxFlow_kernel_7<<<1,1>>>(source1,V,E,d_meta,d_data,d_residual_capacity,d_excess,d_parallel_edge,d_rev_residual_capacity);
 cudaDeviceSynchronize();

  bool flag2 = true; // asst in .cu

  do{
    flag2 = false;

    int hops1 = 0; // asst in .cu
        staticMaxFlow_kernel_10<<<numBlocks, numThreads>>>(source1,sink1,V,E,d_excess,d_level,d_height);
    cudaDeviceSynchronize();

    bool bfsflag1 = false; // asst in .cu

    do{
      bfsflag1 = false;
      cudaMemcpyToSymbol(::bfsflag, &bfsflag1, sizeof(bool), 0, cudaMemcpyHostToDevice);
      staticMaxFlow_kernel_11<<<numBlocks, numThreads>>>(hops1,V,E,d_meta,d_data,d_residual_capacity,d_level,d_rev_residual_capacity,d_height);
      cudaDeviceSynchronize();
      cudaMemcpyFromSymbol(&bfsflag1, ::bfsflag, sizeof(bool), 0, cudaMemcpyDeviceToHost);
      hops1++;
    }while(bfsflag1);
    cudaMemcpyToSymbol(::flag1, &flag2, sizeof(bool), 0, cudaMemcpyHostToDevice);
    staticMaxFlow_kernel_14<<<numBlocks, numThreads>>>(source1,sink1,kernel_cycles1,V,E,d_meta,d_data,d_excess,d_parallel_edge,d_rev_residual_capacity,d_residual_capacity,d_height);
    staticMaxFlow_kernel_17<<<numBlocks, numThreads>>>(source1,sink1,V,E,d_meta,d_data,d_rev_residual_capacity,d_parallel_edge,d_residual_capacity,d_excess,d_height);
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&flag2, ::flag1, sizeof(bool), 0, cudaMemcpyDeviceToHost);

  }while(flag2);

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

  staticMaxFlow(V,E,source,sink,d_residual_capacity,d_rev_residual_capacity,d_excess,kernel_cycles,d_meta,d_data,d_weight, d_parallel_edge,d_level,d_height);

  
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
    Incremental(g,V,E,d_meta,d_data,d_weight,source,sink,d_parallel_edge,d_residual_capacity,d_rev_residual_capacity,d_excess,d_height,d_level,kernel_cycles);
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
  if(src==dest) return;
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


  int _batchSize = batchSize;
  update *d_updateBatch;
  cudaMalloc(&d_updateBatch,sizeof(update)*_batchSize);
  int batchElements = 0;

  int true_cnt = 0;
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


  bool flag1 = true; // asst in .cu
  cudaDeviceSynchronize();
  do{
    flag1 = false;
    int hops = 0; // asst in .cu
    staticMaxFlow_kernel_10<<<numBlocks, numThreads>>>(source0,sink0,V,E,d_excess,d_level,d_height);
    cudaDeviceSynchronize();
    bool bfsflag = false; // asst in .cu
    do{
      bfsflag = false;
      cudaMemcpyToSymbol(::bfsflag, &bfsflag, sizeof(bool), 0, cudaMemcpyHostToDevice);
      staticMaxFlow_kernel_11<<<numBlocks, numThreads>>>(hops,V,E,d_meta,d_data,d_residual_capacity,d_level,d_rev_residual_capacity,d_height);
      cudaDeviceSynchronize();
      cudaMemcpyFromSymbol(&bfsflag, ::bfsflag, sizeof(bool), 0, cudaMemcpyDeviceToHost);
      hops++;
    }while(bfsflag);

    cudaMemcpyToSymbol(::flag1, &flag1, sizeof(bool), 0, cudaMemcpyHostToDevice);
    staticMaxFlow_kernel_14<<<numBlocks, numThreads>>>(source0,sink0,kernel_cycles,V,E,d_meta,d_data,d_excess,d_parallel_edge,d_rev_residual_capacity,d_residual_capacity,d_height);
     staticMaxFlow_kernel_17<<<numBlocks, numThreads>>>(source0,sink0,V,E,d_meta,d_data,d_rev_residual_capacity,d_parallel_edge,d_residual_capacity,d_excess,d_height);
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&flag1, ::flag1, sizeof(bool), 0, cudaMemcpyDeviceToHost);
  }while(flag1);





  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("STATIC UPDATE GPU Time: %.6f ms\n", milliseconds);

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


  return flow;

}

int  main( int  argc, char** argv) {
  char* originalgraph=argv[1];
  char update_filename[MAX_FILENAME_LEN];
char* prefix = argv[2];  // updatefile prefix from command line

  int source = atoi(argv[3]);
  int sink = atoi(argv[4]);
   int i = atoi(argv[5]);
  printf("source:%d sink:%d\n",source,sink);

  graph G1(originalgraph);
    G1.loadGraph();
      int V = G1.num_nodes();
  int E = G1.num_edges();
printf("V:%d\n",V);
printf("E:%d\n",E);


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
        checkCudaError(i*10+1);
        statflow = static_recalculate(G1,updateEdges,updatesSize,source,sink,avg_deg-1);
        checkCudaError(i*10+2);
        if(dynflow!=statflow){
          printf("<<<<<<<<<<<<<<<<<<<<<<<<PANIC NOT MATCHING dynflow:%d statflow:%d>>>>>>>>>>>>>>>>>>>>>>>>>>>\n",dynflow,statflow);
        }
  }

  if(avg_deg>0){
      printf("=======kernel cycles %d =======\n",avg_deg);
  

        dynflow= recalculate_max_flow(G1,updateEdges,updatesSize,source,sink,avg_deg);
        checkCudaError(i*10+3);
        statflow = static_recalculate(G1,updateEdges,updatesSize,source,sink,avg_deg);
        checkCudaError(i*10+4);
        if(dynflow!=statflow){
          printf("<<<<<<<<<<<<<<<<<<<<<<<<PANIC NOT MATCHING dynflow:%d statflow:%d>>>>>>>>>>>>>>>>>>>>>>>>>>>\n",dynflow,statflow);
        }

  }


      printf("=======kernel cycles %d =======\n",avg_deg+1);
  

        dynflow= recalculate_max_flow(G1,updateEdges,updatesSize,source,sink,avg_deg+1);
        checkCudaError(i*10+5);
        statflow = static_recalculate(G1,updateEdges,updatesSize,source,sink,avg_deg+1);
        checkCudaError(i*10+6);
        if(dynflow!=statflow){
          printf("<<<<<<<<<<<<<<<<<<<<<<<<PANIC NOT MATCHING dynflow:%d statflow:%d>>>>>>>>>>>>>>>>>>>>>>>>>>>\n",dynflow,statflow);
        }



  return 0;
}