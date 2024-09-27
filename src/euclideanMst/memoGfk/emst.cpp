#include <tuple>
#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "pargeo/getTime.h"
#include "pargeo/point.h"
#include "kdTree/kdTree.h"
#include "euclideanMst/kruskal.h"
#include "euclideanMst/euclideanMst.h"
#include "wspdFilter.h"
#include "mark.h"

using namespace std;
using namespace parlay;
using namespace pargeo;
using namespace pargeo::emstInternal;

template<int dim>
parlay::sequence<pargeo::wghEdge> pargeo::euclideanMst(parlay::sequence<pargeo::point<dim>> &S) {
  using pointT = point<dim>;
  using nodeT = kdTree::node<dim, point<dim>>;
  using floatT = typename pointT::floatT;
  using pairT = kdTree::wsp<nodeT>;
  using bcpT = tuple<pointT*, pointT*, floatT>;

  if (S.size() < 2) {
    throw std::runtime_error("need more than 2 points");
  }

  bool paraTree = true;

  //nodeT* tree = buildKdt<dim, point<dim>>(S, true, true);
  nodeT* tree = kdTree::build<dim, point<dim>>(S, true, 1);

  floatT rhoLo = -0.1;
  floatT beta = 2;
  size_t numEdges = 0;

  floatT wspdTime = 0;
  floatT kruskalTime = 0;
  floatT markTime = 0;
  edgeUnionFind<long> UF(S.size());

  while (UF.numEdge() < S.size() - 1) {
    floatT rhoHi;
    auto bccps = filterWspdParallel<nodeT>(beta, rhoLo, rhoHi, tree, &UF);
    //auto bccps = filterWspdSerial<nodeT>(beta, rhoLo, rhoHi, tree, &UF);

    numEdges += bccps.size();

    if (bccps.size() <= 0) {
      beta *= 2;
      rhoLo = rhoHi;
      continue;}

    struct wEdge {
      size_t u,v;
      floatT weight;
    };

    auto base = S.data();
    sequence<wEdge> edges = tabulate(bccps.size(), [&](size_t i) {
      auto bcp = bccps[i];
      wEdge e;
      e.u = get<0>(bcp) - base;
      e.v = get<1>(bcp) - base;
      e.weight = get<2>(bcp);
      return e;
    });

    batchKruskal(edges, S.size(), UF);
    
    mark<nodeT, pointT, edgeUnionFind<long>>(tree, &UF, S.data());
    
    beta *= 2;
    rhoLo = rhoHi;
  }

  pargeo::kdTree::del(tree);
  return UF.getEdge();
}


template sequence<wghEdge> pargeo::euclideanMst<2>(sequence<point<2>> &);
template sequence<wghEdge> pargeo::euclideanMst<3>(sequence<point<3>> &);
template sequence<wghEdge> pargeo::euclideanMst<4>(sequence<point<4>> &);
template sequence<wghEdge> pargeo::euclideanMst<5>(sequence<point<5>> &);
template sequence<wghEdge> pargeo::euclideanMst<6>(sequence<point<6>> &);
template sequence<wghEdge> pargeo::euclideanMst<7>(sequence<point<7>> &);
template sequence<wghEdge> pargeo::euclideanMst<8>(sequence<point<8>> &);
template sequence<wghEdge> pargeo::euclideanMst<9>(sequence<point<9>> &);
template sequence<wghEdge> pargeo::euclideanMst<10>(sequence<point<10>> &);
template sequence<wghEdge> pargeo::euclideanMst<11>(sequence<point<11>> &);
template sequence<wghEdge> pargeo::euclideanMst<12>(sequence<point<12>> &);
template sequence<wghEdge> pargeo::euclideanMst<13>(sequence<point<13>> &);
template sequence<wghEdge> pargeo::euclideanMst<14>(sequence<point<14>> &);
template sequence<wghEdge> pargeo::euclideanMst<15>(sequence<point<15>> &);
template sequence<wghEdge> pargeo::euclideanMst<16>(sequence<point<16>> &);
