#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "pargeo/point.h"
#include "pargeo/getTime.h"
#include "pargeo/geometryIO.h"
#include "pargeo/graphIO.h"
#include "pargeo/parseCommandLine.h"
#include "spatialGraph/spatialGraph.h"

using namespace std;
using namespace pargeo;

template<int dim>
void timeGraph(parlay::sequence<pargeo::point<dim>> &P, double b, int rounds, char const *outFile) {
  timer t; t.start();
  for(int i=0; i<rounds; ++i) {
    auto I = pargeo::betaSkeleton<dim>(P, b);
    cout << "round-time = " << t.get_next() << endl;
    if (i == rounds-1 && outFile != NULL) graphIO::writeEdgeSeqToFile(I, outFile);
  }
  t.stop();
}

int main(int argc, char* argv[]) {
  commandLine P(argc,argv,"[-o <outFile>] [-r <rounds>] <inFile>");
  char* iFile = P.getArgument(0);
  double b = P.getOptionDoubleValue("-b",1.0);
  char* oFile = P.getOptionValue("-o");
  int rounds = P.getOptionIntValue("-r",1);

  int dim = readHeader(iFile);

  if (dim == 2) {
    parlay::sequence<pargeo::point<2>> Points = readPointsFromFile<pargeo::point<2>>(iFile);
    timeGraph<2>(Points, b, rounds, oFile);
  } else {
    throw std::runtime_error("unsupported dimensionality");
  }
}
