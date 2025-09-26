#include <iostream>
#include <fstream>
#include <vector>
#include <string>
const std::string OUTPUT_PREFIX = "_p";          // Output file prefix: p1, p2, ..., p10
const std::string OUTPUT_PREFIX2 = "_del";  
int main(int argc, char* argv[]) {
   
    const char* input_file = argv[1];
    const char* output_folder = argv[2];
    int x = atoi(argv[3]);
    int p = atoi(argv[4]);

    std::ifstream fin(input_file);
    if (!fin.is_open()) {
        std::cerr << "❌ Error: Cannot open input file: " << input_file << "\n";
        return 1;
    }

    std::string out_filename = output_folder+ OUTPUT_PREFIX + std::to_string(p) +".txt";

    std::ofstream fout(out_filename);
    if (!fout.is_open()) {
        std::cerr << "❌ Error: Cannot open input file: " << out_filename << "\n";
        return 1;
    }


    std::string line;
    int line_count = 0;
    char a; int u,v,w;
    fout<<x*p<<"\n";
        for(int l = 0;l<(x*p);l++){

            fin >>v>>u>>w;
            fout<<v<<" " <<u<<" "<<w<< "\n";

        }
    
    // Close all files
    fin.close();
    fout.close();

    // std::cout << "✅ Successfully created p"<<p<<".\n";
    return 0;
}
