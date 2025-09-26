#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

using namespace std;

// Generate a random number within a range using a shared generator
int getRandomChange(mt19937 &gen, int min, int max) {
    if (max < min) return 0;
    uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}

void processFile(const string& inputFile, const string& outputFile, double percentage,
                 double increasePercentage, int source, int sink) {
    ifstream inFile(inputFile);
    ofstream outFile(outputFile);

    if (!inFile.is_open()) {
        cerr << "Error opening input file!" << endl;
        return;
    }

    if (!outFile.is_open()) {
        cerr << "Error opening output file!" <<outputFile<< endl;
        return;
    }

    string line;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> probDist(0.0, 1.0);

    size_t totalEdges = 0;
    size_t modifiedEdges = 0;

    while (getline(inFile, line)) {
        istringstream iss(line);
        int u, v, w;
        if (!(iss >> u >> v >> w)) continue;

        totalEdges++;

        bool isSpecialEdge = (u == source || v == sink);
        bool shouldModify = (probDist(gen) < percentage / 100.0);
        if (isSpecialEdge) shouldModify = shouldModify || (probDist(gen) < percentage / 10.0);
        if (u == v) shouldModify = false;

        if (shouldModify) {
            bool increase = (probDist(gen) < increasePercentage / 100.0);
            if (increase) {
                int increaseAmount = getRandomChange(gen, 1, max(1, 101 - w));
                w += increaseAmount;
            } else {
                int decreaseAmount = getRandomChange(gen, 1, max(1, w));
                w -= decreaseAmount;
            }

            if (w >= 0 && w <= 101) {
                modifiedEdges++;
                outFile << u << " " << v << " " << w << "\n";
            }
        }

        // Optional progress logging
        if (totalEdges % 1000000 == 0) {
            cerr << "Processed " << totalEdges << " edges..." << endl;
        }
    }

    inFile.close();
    outFile.close();

    double modifiedPercentage = (totalEdges > 0) ? (modifiedEdges * 100.0 / totalEdges) : 0.0;
    cout<< "total size of graph: "<< totalEdges <<" batch size: " <<totalEdges/100<<endl;
    cout << "Edges modified: " << modifiedEdges << " (" << modifiedPercentage << "%)" << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        cerr << "Usage: " << argv[0]
             << " <inputFile> <outputFile> <percentage> <increasePercentage> <source> <sink>" << endl;
        return 1;
    }

    string inputFile = argv[1];
    string outputFile = argv[2];
    double percentage = stod(argv[3]);
    double increasePercentage = stod(argv[4]);
    int source = atoi(argv[5]);
    int sink = atoi(argv[6]);

    processFile(inputFile, outputFile, percentage, increasePercentage, source, sink);
    return 0;
}
