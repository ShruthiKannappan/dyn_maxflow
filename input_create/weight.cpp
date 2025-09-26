#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <string>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input_edge_list.txt output_weighted_edge_list.txt\n";
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        std::cerr << "Error opening input file: " << input_file << "\n";
        return 1;
    }

    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "Error opening output file: " << output_file << "\n";
        return 1;
    }

    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(1, 100);

    std::string line;
    size_t line_count = 0;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '%') continue;

        std::istringstream iss(line);
        int u, v;
        // long long unsigned t;
        if (!(iss >> u >> v)) {
            std::cerr << "Skipping invalid line: " << line << "\n";
            continue;
        }
        int weight = dist(rng);
        outfile << u << " " << v << " " << weight << "\n";

        if (++line_count % 1000000 == 0) {
            std::cout << "Processed " << line_count << " edges...\n";
        }
    }

    std::cout << "Finished processing " << line_count << " edges.\n";

    infile.close();
    outfile.close();

    return 0;
}
