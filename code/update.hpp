#include <vector>
#include <fstream>
#include <sstream>
#include <string.h>

class update
{
public:
  char type;
  int32_t source;
  int32_t destination;
  int32_t weight;
};

std::vector<update> parseUpdateFile(char *updateFile)
{
  std::vector<update> updates;
  std::ifstream infile;
  infile.open(updateFile);
  if (!infile.is_open()) {
    printf( "Error: Failed to open file %s for reading.\n", updateFile);
    return updates; // or handle error appropriately
} 
  std::string line;

  int line_number = 0;
  while (std::getline(infile, line))
  {
    std::stringstream ss(line);
  
    char type;
    int32_t source;
    int32_t destination;
    int32_t wt;

    if (!(ss >> type >> source >> destination >> wt)) {
      printf("Warning: Malformed line%d in update file Halting.\n",line_number);break;
      continue;
    }

    update u;
    u.type = type;
    u.source = source;
    u.destination = destination;
    u.weight = wt;
    if(u.source==u.destination) continue;
    updates.push_back(u);
  }
  return updates;
}

std::vector<update> loadUpdateFile(char *updateFile)
{
  
    std::ifstream fin(updateFile, std::ios::binary);
    if (!fin) {
        std::cerr << "Error: cannot open " << updateFile << " for reading\n";
        std::vector<update> updates;
        return updates ;
    }
    int batchsize;
    fin.read(reinterpret_cast<char*>(&batchsize), sizeof(int32_t));
    int * src, *dst, *weight;
    src = (int32_t*)malloc(sizeof(int32_t)*(batchsize));
    dst = (int32_t*)malloc(sizeof(int32_t)*(batchsize));
    weight = (int32_t*)malloc(sizeof(int32_t)*(batchsize));

    fin.read(reinterpret_cast<char*>(src), (batchsize) * sizeof(int32_t));
    fin.read(reinterpret_cast<char*>(dst), (batchsize) * sizeof(int32_t));
    fin.read(reinterpret_cast<char*>(weight), (batchsize) * sizeof(int32_t));
     fin.close();
  std::vector<update> updates(batchsize);


  for(int i = 0;i<batchsize;i++)
  {
    update u;
    u.type = 'a';
    u.source = src[i];
    u.destination = dst[i];
    u.weight = weight[i];
    if(u.source==u.destination) continue;
    updates.push_back(u);
  }
  free(src);
  free(dst);
  free(weight);
  return updates;
}