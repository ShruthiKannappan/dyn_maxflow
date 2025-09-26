#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

struct Edge {
    int to;
    int weight;
    int parallel_ind;
};

struct TTuple{
    int u, v,w;
};

// ----------------- Build CSR and Write BIN -----------------
void writeCSR(const std::vector<std::vector<Edge>> &adj, const std::string &filename) {
    int n = adj.size();
    int m = 0;
    for (const auto &nbrs : adj) m += nbrs.size();

    std::vector<int> index(n + 1, 0);
    std::vector<int> edges;
    std::vector<int> weights;
    std:: vector<int> parallel_edges;

    edges.reserve(m);
    weights.reserve(m);
    weights.reserve(m);
    int pos = 0;
    for (int u = 0; u < n; u++) {
        index[u] = pos;
        for (const auto &e : adj[u]) {
            edges.push_back(e.to);
            weights.push_back(e.weight);
            pos++;
        }
    }
    index[n] = pos;
    for(int u = 0;u<n;u++){
        for (const auto &e : adj[u]) {
            parallel_edges.push_back(e.parallel_ind+index[e.to]);
        }
    }


    std::ofstream fout(filename, std::ios::binary);
    if (!fout) {
        std::cerr << "Error: cannot open " << filename << " for writing\n";
        return;
    }


    fout.write(reinterpret_cast<const char*>(&n), sizeof(int));
    fout.write(reinterpret_cast<const char*>(&m), sizeof(int));
    fout.write(reinterpret_cast<const char*>(index.data()), (n + 1) * sizeof(int));
    fout.write(reinterpret_cast<const char*>(edges.data()), m * sizeof(int));
    fout.write(reinterpret_cast<const char*>(weights.data()), m * sizeof(int));
    fout.write(reinterpret_cast<const char*>(parallel_edges.data()), m * sizeof(int));
    fout.close();

    std::cout << "CSR written to " << filename << " (" << n << " nodes, " << m << " edges)\n";
}

// ----------------- Read Edge List -----------------
void readEdgeList(const std::string &filename, std::vector<std::vector<Edge>> &adj) {
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Error: cannot open " << filename << " for reading\n";
        return;
    }

    int maxNode = -1;
    std::vector<TTuple> edges;

    std::string line;
    edges.reserve(1000000); // reserve some space for efficiency

    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue; // skip comments
        std::istringstream iss(line);
        int u, v,w;
            iss >> u >> v >> w;
        if(u==v) continue;
        edges.push_back({u:u,v:v,w:w});
        if(maxNode < u) maxNode = u;
        if(maxNode<v) maxNode = v;
    }
    fin.close();

    int n = maxNode + 1;
    adj.assign(n, {});

    for (auto &e : edges) {
        int u = e.u;
        int v = e.v;
        int w = e.w;
        int ind_forward = -1;
        for(int i = 0;i<adj[u].size();i++){
            if(adj[u][i].to==v){
                ind_forward = i;
                break;
            }
        }
        if(ind_forward == -1) {adj[u].push_back({to:v ,weight:w,parallel_ind:-1});ind_forward = adj[u].size()-1;}
        else adj[u][ind_forward].weight+= w;

        int ind_rev = -1;
        for(int i = 0;i<adj[v].size();i++){
            if(adj[v][i].to == u){
                ind_rev = i;
                break;
            }
        }
        if(ind_rev==-1)   {adj[v].push_back({to:u ,weight:0,parallel_ind:-1});ind_rev = adj[v].size()-1;}
        adj[u][ind_forward].parallel_ind = ind_rev;
        adj[v][ind_rev].parallel_ind = ind_forward;
    }
    
    

    std::cout << "Read " << edges.size() << " edges, " << n << " nodes\n";
}

// ----------------- MAIN -----------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.txt output.bin \n";
        return 1;
    }

    std::string infile = argv[1];
    std::string outfile = argv[2];

    std::vector<std::vector<Edge>> adj;
    readEdgeList(infile, adj);
    writeCSR(adj, outfile);

    return 0;
}
