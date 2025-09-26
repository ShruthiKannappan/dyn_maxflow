#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>




// ----------------- Read Edge List ----------------
void readEdgeList(const std::string &filename, const std::string &outfilename) {
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Error: cannot open " << filename << " for reading\n";
        return;
    }
    int batchsize;
    std::string line;
    std::getline(fin, line);
    std::istringstream iss2(line);
    iss2>> batchsize;
    int *src, *dst, *weight;
    src = (int32_t*) malloc(sizeof(int32_t)*batchsize);
    dst = (int32_t*) malloc(sizeof(int32_t)*batchsize);
    weight = (int32_t*) malloc(sizeof(int32_t)*batchsize);
    int final_ptr = 0;
    while (std::getline(fin, line)) {
        if(final_ptr>=batchsize) break;
        std::istringstream iss(line);
        int u, v,w;
        iss >> u >> v >> w;
        src[final_ptr] = u;
        dst[final_ptr] = v;
        weight[final_ptr] = w;
        final_ptr++;
    }
    fin.close();

    std::ofstream fout(outfilename, std::ios::binary);
    if (!fout) {
        std::cerr << "Error: cannot open " << outfilename << " for writing\n";
        return;
    }


    fout.write(reinterpret_cast<const char*>(&batchsize), sizeof(int));
    fout.write(reinterpret_cast<const char*>(src), (batchsize) * sizeof(int));
    fout.write(reinterpret_cast<const char*>(dst), batchsize * sizeof(int));
    fout.write(reinterpret_cast<const char*>(weight), batchsize * sizeof(int));
    fout.close();

}

// ----------------- MAIN -----------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.txt output.bin \n";
        return 1;
    }

    std::string infile = argv[1];
    std::string outfile = argv[2];

    readEdgeList(infile, outfile);
    return 0;
}
