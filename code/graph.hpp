#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <string.h>
#include"update.hpp"

class edge
{
public:
  int32_t source;
  int32_t destination;
  int32_t weight;
};

class graph
{
private:
  int32_t nodesTotal;                         //// total number of nodes present in the graph
  int32_t edgesTotal;                         //// total number of edges present in CSR/CSC
  int32_t *edgeLen;                           //// for storing edge weights (CSR)
  char *filePath;                             //// path to the input file
  std::vector<std::vector<edge>>edges; //// stores out-neighbours. key=node, value=out-neighbours
  
public:
  int32_t *indexofNodes;          //// stores offsets of edges in CSR
  int32_t *edgeList;              //// stores edges in CSR
  int32_t *parallel_edge;

  graph(char *file)
  {
    filePath = file;
    nodesTotal = 0;
    edgesTotal = 0;
    edgeLen = NULL;
    indexofNodes = NULL;
    edgeList = NULL;
  }

  void find_total_nodes(){
        std::ifstream infile(filePath);
    if (!infile.is_open()) {
        throw std::runtime_error("Error: could not open file.");
    }

    int32_t src, dst, w;
    int32_t maxNodeId = -1;

    // ---- Pass 1: count degrees ----
    std::vector<int32_t> degree; // big enough for nodes
    edgesTotal = 0;
    while (infile >> src >> dst >> w) {
        if (src == dst) continue; // skip self-loops
        maxNodeId = std::max({maxNodeId, src, dst});

        if ((int)degree.size() <= maxNodeId) {
            degree.resize(maxNodeId + 1, 0);
        }
        degree[src]++; 
        degree[dst]++; // for reverse edge
        edgesTotal+=2;
    }
    
    infile.close();

    nodesTotal = maxNodeId;
    edges.clear();
    edges.resize(nodesTotal + 1);
    for (int i = 0; i <= nodesTotal; i++) {
        edges[i].resize(degree[i]); // no reallocation now
        degree[i] = 0;
    }

    std::ifstream infile2(filePath);
    if (!infile2.is_open()) {
        throw std::runtime_error("Error: could not open file.");
    }
    while (infile2 >> src >> dst >> w) {
        if (src == dst) continue; // skip self-loops
        int ind = degree[src];
        int ind_rev = degree[dst];
        edges[src][ind].source = src;
        edges[src][ind].destination = dst;
        edges[src][ind].weight = w;
        edges[dst][ind_rev].destination = src;
        edges[dst][ind_rev].weight = 0;
        edges[dst][ind_rev].source = dst;
        degree[dst]++;
        degree[src]++;
    }
    
    infile2.close();
  }

  void parseEdges(){
    find_total_nodes();
    int edge_no = 0;
        indexofNodes =  (int32_t*) malloc(sizeof(int32_t)*(nodesTotal+2));
    int32_t * temp_edgeList = (int32_t*) malloc(sizeof(int32_t)*edgesTotal);
    int32_t * temp_edgeLen =(int32_t*) malloc(sizeof(int32_t)*edgesTotal);
    edgesTotal = 0;
    for(int i = 0;i<=nodesTotal;i++){
      indexofNodes[i] = edge_no;
      auto curList = &edges[i];
      sort(curList->begin(),curList->end(), [](const edge &e1, const edge &e2)
           {
             return e1.destination < e2.destination;
             });
      auto itr = curList->begin();
      while(itr!=curList->end()){
        if(edge_no==indexofNodes[i]||temp_edgeList[edge_no-1]!=itr->destination){
          int nbr = itr->destination;
          temp_edgeList[edge_no] = nbr;
          temp_edgeLen[edge_no] = itr->weight;
          edge_no++;
        }
        else{
          temp_edgeLen[edge_no-1]+=itr->weight;          
        }
        itr++;
      }

    }
    indexofNodes[nodesTotal+1] = edge_no;
    edgesTotal = edge_no;
    edgeList = (int32_t*) malloc(sizeof(int32_t)*edgesTotal);
    edgeLen = (int32_t*) malloc(sizeof(int32_t)*edgesTotal);
    parallel_edge = (int32_t*) malloc(sizeof(int32_t)*edgesTotal);
    for(int i = 0;i<=nodesTotal;i++){
      for(int e = indexofNodes[i];e<indexofNodes[i+1];e++){
        int nbr = temp_edgeList[e];
        edgeList[e] = nbr;
        edgeLen[e] = temp_edgeLen[e];
        if(nbr<i){
          int rev_index = -1;
          for(int j = indexofNodes[nbr];j<indexofNodes[nbr+1];j++){
            if(edgeList[j]==i){
              rev_index = j;
              break;
            }
          }
          if(rev_index==-1){
            printf("panic error parsing\n");
            exit(0);
          }
          parallel_edge[rev_index] = e;
          parallel_edge[e] = rev_index;
        }
      }  
      
    }
    free(temp_edgeList);
    free(temp_edgeLen);
    edges.clear();
  }


  void parseGraph()
  {
    parseEdges();    
  }
 
  int *getEdgeLen()
  {
    return edgeLen;
  }

  int num_nodes()
  {
    return nodesTotal;
  }

  int num_edges()
  {
      return edgesTotal;
  }

  int num_edges_CSR()
  {
    return edgesTotal;
  }

std::vector<update> parseUpdates(char *updateFile)
{
    std::vector<update> update_vec;
    try {
        if (updateFile == nullptr) {
            throw std::invalid_argument("Error: updateFile path is null.");
        }
        update_vec = loadUpdateFile(updateFile);
    } catch (const std::exception& ex) {
        std::cerr << "Exception caught in parseUpdates: " << ex.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error occurred in parseUpdates." << std::endl;
    }

    return update_vec;
}

void loadGraph(){
      std::ifstream fin(filePath, std::ios::binary);
    if (!fin) {
        std::cerr << "Error: cannot open " << filePath << " for reading\n";
        return ;
    }
        // Read n, m
    fin.read(reinterpret_cast<char*>(&nodesTotal), sizeof(int32_t));
    fin.read(reinterpret_cast<char*>(&edgesTotal), sizeof(int32_t));

    indexofNodes = (int32_t*)malloc(sizeof(int32_t)*(nodesTotal+1));
    edgeList = (int32_t*)malloc(sizeof(int32_t)*edgesTotal);
    edgeLen = (int32_t*)malloc(sizeof(int32_t)*edgesTotal);
    parallel_edge = (int32_t*)malloc(sizeof(int32_t)*edgesTotal);

    fin.read(reinterpret_cast<char*>(indexofNodes), (nodesTotal+ 1) * sizeof(int32_t));
    fin.read(reinterpret_cast<char*>(edgeList), (edgesTotal) * sizeof(int32_t));
    fin.read(reinterpret_cast<char*>(edgeLen), (edgesTotal) * sizeof(int32_t));
    fin.read(reinterpret_cast<char*>(parallel_edge), (edgesTotal) * sizeof(int32_t));
     fin.close();
    return ;
}

~graph() {
    free(edgeLen);
    free(edgeList);
    free(indexofNodes);
    free(parallel_edge);
}
 

};