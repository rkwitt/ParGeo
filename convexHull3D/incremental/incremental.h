// This code is part of the Pargeo Library
// Copyright (c) 2020 Yiqiu Wang and the Pargeo Team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

//#define WRITE // Write to file, visualize using python3 plot.py
//#define VERBOSE
#define SEQ_SPLIT

#ifdef WRITE
#include <iostream>
#include <fstream>
#endif

#include <atomic>
#include <vector>
#include <assert.h>
#include <stack>
#include <tuple>
#include "parlay/sequence.h"
#include "geometry/point.h"
#include "common/algebra.h"
#include "common/get_time.h"
#include "hull.h"

using namespace parlay;
using namespace parlay::internal;
using pointT = pargeo::fpoint<3>;
static const typename pointT::floatT numericKnob = 1e-5;

////////////////////////////
// Primitives
////////////////////////////

/* Signed volume of an oriented tetrahedron (example below is positive).
     d
     o
    /|\
   / | o b
  o--o/
   a   c
*/
template <class pt>
inline typename pt::floatT signedVolume(pt a, pt b, pt c, pt d) {
  return (a-d).dot(crossProduct3d(b-a, c-a))/6;
}

template<class facetT, class vertexT>
bool visible(facetT* f, vertexT p) {
  if (signedVolume(f->a, f->b, f->c, p) > numericKnob)
    return true;
  else
    return false;
}

////////////////////////////
// Facet and vertex with attributes
////////////////////////////

template <class pt> struct linkedFacet3d;

struct vertexAtt;

using vertex3d = pargeo::_point<3, pointT::floatT, pointT::floatT, vertexAtt>;

struct vertexAtt {
  linkedFacet3d<vertex3d> *seeFacet;
  vertexAtt() {}
};

static std::ostream& operator<<(std::ostream& os, const vertex3d v) {
  for (int i=0; i<v.dim; ++i)
    os << v.x[i] << " ";
  return os;
}

template <class pt>
struct linkedFacet3d {
  vertex3d a, b, c;
  linkedFacet3d<pt> *abFacet;
  linkedFacet3d<pt> *bcFacet;
  linkedFacet3d<pt> *caFacet;

  // Stores the minimum memory address of the seeFacet of the reserving vertices
  std::atomic<size_t> reservation; // todo (not always used but has little impact on speed)

#ifdef SEQ_SPLIT

  vector<vertex3d> *seeList;
  vertex3d& at(size_t i) { return seeList->at(i); }
  size_t size() { return seeList->size(); }
  void push_back(vertex3d v) { seeList->push_back(v); }

#else

  slice<vertex3d*, vertex3d*> seeList;
  vertex3d& at(size_t i) { return seeList[i]; }
  size_t size() { return seeList.size(); }

#endif

  linkedFacet3d(vertex3d _a, vertex3d _b, vertex3d _c): a(_a), b(_b), c(_c) {
    if (pargeo::determinant3by3(a, b, c) > 0)
      swap(b, c);

#ifdef SEQ_SPLIT
    seeList = new vector<vertex3d>();
#endif

    reservation = -1; // (unsigned)
  }

  ~linkedFacet3d() {
#ifdef SEQ_SPLIT
    delete seeList;
#endif
  }
};

template<class pt>
static std::ostream& operator<<(std::ostream& os, const linkedFacet3d<pt> v) {
  os << "(" << v.a << "," << v.b << "," << v.c << ")";
  return os;
}

// a, b, c can be input in any order
linkedFacet3d<vertex3d>* makeFacet(vertex3d a, vertex3d b, vertex3d c) {
  auto f = new linkedFacet3d<vertex3d>(a, b, c);
  return f;
}

// Link f with ab, bc, ca; the edge matching is automatic -- input in any order
template<class fc>
void linkFacet(fc* f, fc* ab, fc* bc, fc* ca) {
  fc* F[3]; F[0]=ab; F[1]=bc; F[2]=ca;
  auto findFacet = [&](vertex3d v1, vertex3d v2) {
		     for(int i=0; i<3; ++i) {
		       if ((F[i]->a==v1 && F[i]->b==v2) || (F[i]->b==v1 && F[i]->a==v2) ||
			   (F[i]->b==v1 && F[i]->c==v2) || (F[i]->c==v1 && F[i]->b==v2) ||
			   (F[i]->c==v1 && F[i]->a==v2) || (F[i]->a==v1 && F[i]->c==v2)) {
			 return F[i];
		       }
		     }
		     cout << "Error, facet linking failure, abort." << endl;
		     abort();
		   };

  auto linkEdge = [&](fc* f1, fc* f2, vertex3d v1, vertex3d v2) {
		    if ( (f2->a==v1 && f2->b==v2) || (f2->a==v2 && f2->b==v1) )
		      f2->abFacet = f1;
		    else if ( (f2->b==v1 && f2->c==v2) || (f2->b==v2 && f2->c==v1) )
		      f2->bcFacet = f1;
		    else if ( (f2->c==v1 && f2->a==v2) || (f2->c==v2 && f2->a==v1) )
		      f2->caFacet = f1;
		    else {
		      cout << "Error, edge linking failure, abort." << endl;
		      abort();
		    }
		  };

  f->abFacet = findFacet(f->a, f->b);
  linkEdge(f, f->abFacet, f->a, f->b);
  f->bcFacet = findFacet(f->b, f->c);
  linkEdge(f, f->bcFacet, f->b, f->c);
  f->caFacet = findFacet(f->c, f->a);
  linkEdge(f, f->caFacet, f->c, f->a);
}

////////////////////////////
// Hull structure
////////////////////////////

/* An edge structure for advancing direction in a traversal

   ff: forward facet to advance to
   fb: parent facet from which we come

       o
      / \ --->ff
   a o---o b
      \ / --->fb
       o
*/
template <class facetT, class vertexT>
struct _edge {
  vertexT a, b;
  facetT* ff;
  facetT* fb;

  _edge(vertexT _a, vertexT _b, facetT* _ff, facetT* _fb) {
    a = _a; b = _b; ff = _ff; fb = _fb;}

  friend bool operator==(_edge e1, _edge e2) {
    return e1.a==e2.a && e1.b==e2.b;}//todo
};

/* A structure that maintains and traverses the facets
 */
template <class facetT, class vertexT>
struct _hull {
  facetT* H;
  //sequence<vertexT>* Q;

  _hull(facetT* _H) {
    H = _H;
  }

  /* Depth-first hull traversal (no facet repeat)
  */
  template <class F, class G, class H>
  void dfsFacet(facetT* start, F& fVisit, G& fDo, H& fStop) {
    sequence<facetT*> V;
    auto mark = [&](facetT* f) {V.push_back(f);};
    auto visited = [&](facetT* f) {
		     for (auto g: V) {
		       if (f == g) return true;
		     }
		     return false;};

    stack<_edge<facetT, vertexT>> S;

    S.emplace(start->b, start->a, start, nullptr);

    while (S.size() > 0) {
      _edge e = S.top(); S.pop();
      if (!visited(e.ff) && fVisit(e)) {
	fDo(e);
	mark(e.ff);

	S.emplace(e.ff->a, e.ff->b, e.ff->abFacet, e.ff);
	S.emplace(e.ff->c, e.ff->a, e.ff->caFacet, e.ff);
	S.emplace(e.ff->b, e.ff->c, e.ff->bcFacet, e.ff);
      }
      if (fStop()) break;
    }
  }

  /* Clockwise, depth-first hull traversal, visits each oriented edge once
     - (facetT*) start: starting facet
     - (func _edge -> bool) fVisit : whether to visit the target facet of an advancing edge
     - (func _edge -> void) fDo : what do to with the facet in the advancing edge
     - (func void -> bool)  fStop : whether to stop
  */
  template <class F, class G, class H>
  void dfsEdge(facetT* start, F& fVisit, G& fDo, H& fStop) {
    using edgeT = _edge<facetT, vertexT>;

    sequence<edgeT> V;
    auto mark = [&](edgeT f) {V.push_back(f);};
    auto visited = [&](edgeT f) {
		     for (auto g: V) {
		       if (f == g) return true;
		     }
		     return false;};

    stack<edgeT> S;

    /* Create initial advancing edge (b,a)
         o start->c
        / \ ---> start
       o---o
start->b  start->a
    */
    S.emplace(start->b, start->a, start, nullptr);
    while (S.size() > 0) {
      _edge e = S.top(); S.pop();
      if (!visited(e) && fVisit(e)) {
	fDo(e);
	mark(e);
	/* Push in ccw order, pop in cw order; start from (e.ff.a, e.ff.b)
	   e.ff.b
	     o
	    / \ ---> e.ff
	   o---o
e.b==e.ff.a    e.a==e.ff.c
	*/
	if (e.ff->a == e.b) {
	  S.emplace(e.ff->a, e.ff->b, e.ff->abFacet, e.ff);
	  S.emplace(e.ff->c, e.ff->a, e.ff->caFacet, e.ff);
	  S.emplace(e.ff->b, e.ff->c, e.ff->bcFacet, e.ff);
	} else if (e.ff->b == e.b) {
	  S.emplace(e.ff->b, e.ff->c, e.ff->bcFacet, e.ff);
	  S.emplace(e.ff->a, e.ff->b, e.ff->abFacet, e.ff);
	  S.emplace(e.ff->c, e.ff->a, e.ff->caFacet, e.ff);
	} else if (e.ff->c == e.b) {
	  S.emplace(e.ff->c, e.ff->a, e.ff->caFacet, e.ff);
	  S.emplace(e.ff->b, e.ff->c, e.ff->bcFacet, e.ff);
	  S.emplace(e.ff->a, e.ff->b, e.ff->abFacet, e.ff);
	}
      }
      if (fStop()) break;
    }
  }

  size_t hullSize(facetT* start=nullptr) {
    size_t s = 0;
    auto fVisit = [&](_edge<facetT, vertexT> e) { return true;};
    auto fDo = [&](_edge<facetT, vertexT> e) { s++;};
    auto fStop = [&]() { return false;};
    if (start) dfsFacet(start, fVisit, fDo, fStop);
    else dfsFacet(H, fVisit, fDo, fStop);
    return s;
  }

  // Also checks the hull integrity
  void printHull(facetT* start=nullptr, bool checker=true) {
    if (checker) printHull(start, false);
    size_t hs = hullSize();
    auto fVisit = [&](_edge<facetT, vertexT> e) { return true;};
    auto fDo = [&](_edge<facetT, vertexT> e) {
		 if (checker && hullSize(e.ff) != hs) {
		   cout << " ..." << endl;
		   cout << "Error, inconsistency detected, abort" << endl;
		   cout << "Erroneous hull size = " << hullSize(e.ff) << endl;
		   abort();
		 }
		 if (!checker) cout << *e.ff << " ";
	       };
    auto fStop = [&]() { return false;};

    if (!checker) cout << "Hull DFS (" << hs << ")  = ";
    if (start) dfsFacet(start, fVisit, fDo, fStop);
    else dfsFacet(H, fVisit, fDo, fStop);
    if (!checker) cout << endl;
  }

#ifdef WRITE
  void writeHull() {
    using edgeT = _edge<facetT, vertexT>;
    ofstream myfile;
    myfile.open("hull.txt", std::ofstream::trunc);
    auto fVisit = [&](edgeT e) { return true;};
    auto fDo = [&](edgeT e) {
		 myfile << e.ff->a << endl;
		 myfile << e.ff->b << endl;
		 myfile << e.ff->c << endl;
	       };
    auto fStop = [&]() { return false;};
    dfsFacet(H, fVisit, fDo, fStop);
    myfile.close();
  }
#endif
};

////////////////////////////
// Conflict graph
////////////////////////////

template <class facetT, class vertexT>
class conflictList {

public:
  _hull<facetT, vertexT> *context;

  conflictList(_hull<facetT, vertexT> *_c): context(_c) {};

  /* Choose a random outside vertex visible to some facet
   */
  vertexT randomApex() {
    if (context->H->size() > 0) return context->H->at(0);
    vertexT apex = vertexT();
    auto fVisit = [&](_edge<facetT, vertexT> e) {return true;};
    auto fDo = [&](_edge<facetT, vertexT> e) {
		 if (e.ff->size() > 0)
		   apex = e.ff->at(0);};
    auto fStop = [&]() {
		   if (!apex.isEmpty()) return true;
		   else return false;};
    context->dfsFacet(context->H, fVisit, fDo, fStop);
    return apex;
  }

  /* Choose the furthest outside vertex visible to some facets
     look specifices the maxmimum number (adjacent) facets to try
   */
  vertexT furthestApex(size_t look=1) {
    typename vertexT::floatT m = numericKnob;
    vertexT apex = vertexT();

    auto find = [&](facetT* f) {
		  for (size_t i=0; i<f->size(); ++i) {
		    auto m2 = signedVolume(f->a, f->b, f->c, f->at(i));
		    if (m2 > m) {
		      m = m2; apex = f->at(i);
		    }
		  }
		};

    auto fVisit = [&](_edge<facetT, vertexT> e) {return true;};
    auto fDo = [&](_edge<facetT, vertexT> e) {
		 if (e.ff->size() > 0) {
		   look--;
		   find(e.ff);
		 }
	       };
    auto fStop = [&]() {
		   if (look <= 0) return true;
		   else return false;};
    context->dfsFacet(context->H, fVisit, fDo, fStop);
    return apex;
  }

  // If n not supplied, find furthest apexes of all facets
  vector<vertexT> furthestApexes(size_t n=-1) {
    size_t found = 0;
    vector<vertexT> apexes;

    auto find = [&](facetT* f) {
		  typename vertexT::floatT m = numericKnob;
		  auto apex = vertexT();

		  for (size_t i=0; i<f->size(); ++i) {
		    auto m2 = signedVolume(f->a, f->b, f->c, f->at(i));
		    if (m2 > m) {
		      m = m2; apex = f->at(i);
		    }
		  }
		  if (!apex.isEmpty()) apexes.push_back(apex);
		};

    auto fVisit = [&](_edge<facetT, vertexT> e) {return true;};
    auto fDo = [&](_edge<facetT, vertexT> e) {
		 if (e.ff->size() > 0) find(e.ff);
	       };
    auto fStop = [&]() {return apexes.size() >= n;};
    context->dfsFacet(context->H, fVisit, fDo, fStop);
    return apexes;
  }

  /* Compute a frontier of edges in the clockwise order
      and facets to delete
   */
  tuple<sequence<_edge<facetT, vertexT>>*, sequence<facetT*>*> computeFrontier(vertexT apex) {
    facetT* fVisible = apex.attribute.seeFacet;

    auto frontier = new sequence<_edge<facetT, vertexT>>();
    auto facets = new sequence<facetT*>();
    auto facetVisited = [&](facetT* f) {
			  for (size_t i=0; i<facets->size(); ++i) {
			    if (f == facets->at(i)) return true;
			  }
			  return false;
			};

    auto fVisit = [&](_edge<facetT, vertexT> e) {
		    // Visit the facet as long as the parent facet is visible to the apex
		    // e.fb == nullptr for the starting facet (whose parent is nullptr, see dfsEdge(...))
		    if (e.fb == nullptr || visible(e.fb, apex))
		      return true;
		    else
		      return false;
  		  };

    auto fDo = [&](_edge<facetT, vertexT> e) {
		 // Include the facet for deletion if visible
		 bool seeff = visible(e.ff, apex);
		 if ((seeff || e.fb == nullptr) && !facetVisited(e.ff))
		   facets->push_back(e.ff);

		 if (e.fb == nullptr) return; // Stop for the starting facet

		 // Include an edge joining a visible and an invisible facet as frontier
		 bool seefb = visible(e.fb, apex);
		 if (seefb && !seeff) {
  		   frontier->emplace_back(e.a, e.b, e.ff, e.fb);
		 }
  	       };
    auto fStop = [&](){ return false;};

    context->dfsEdge(apex.attribute.seeFacet, fVisit, fDo, fStop);
    return make_tuple(frontier, facets);
  }

  /* Compute a frontier of edges in the clockwise order
      and facets to delete

     Meanwhile reserve the facet using min(apex.attribute.seeFacet*)
   */
  tuple<sequence<_edge<facetT, vertexT>>*, sequence<facetT*>*> computeFrontierAndReserve(vertexT apex) {
    facetT* fVisible = apex.attribute.seeFacet;

    auto frontier = new sequence<_edge<facetT, vertexT>>();
    auto facets = new sequence<facetT*>();
    auto facetVisited = [&](facetT* f) {
			  for (size_t i=0; i<facets->size(); ++i) {
			    if (f == facets->at(i)) return true;
			  }
			  return false;
			};

    auto fVisit = [&](_edge<facetT, vertexT> e) {
		    // Visit the facet as long as the parent facet is visible to the apex
		    // e.fb == nullptr for the starting facet (whose parent is nullptr, see dfsEdge(...))
		    if (e.fb == nullptr || visible(e.fb, apex))
		      return true;
		    else
		      return false;
		  };

    auto fDo = [&](_edge<facetT, vertexT> e) {
		 // Include the facet for deletion if visible
		 // Also reserve the facet
		 bool seeff = visible(e.ff, apex);
		 if ((seeff || e.fb == nullptr) && !facetVisited(e.ff)) {
		   facets->push_back(e.ff);
		   parlay::write_min(&(e.ff->reservation), size_t(apex.attribute.seeFacet), std::less<size_t>());
		 }

		 if (e.fb == nullptr) return; // Stop for the starting facet

		 // Include an edge joining a visible and an invisible facet as frontier
		 // Also reserve invisible facet adjacent to a visible one
		 bool seefb = visible(e.fb, apex);
		 if (seefb && !seeff) {
		   frontier->emplace_back(e.a, e.b, e.ff, e.fb);
		   parlay::write_min(&(e.ff->reservation), size_t(apex.attribute.seeFacet), std::less<size_t>());
		 }
	       };
    auto fStop = [&](){ return false;};

    context->dfsEdge(apex.attribute.seeFacet, fVisit, fDo, fStop);
    return make_tuple(frontier, facets);
  }

  bool confirmReservation(vertexT apex, slice<facetT**, facetT**> toDelete) {
    bool ok = true;
    for (auto f: toDelete) {
      if (f->reservation != size_t(apex.attribute.seeFacet) ||
	  f->abFacet->reservation != size_t(apex.attribute.seeFacet) ||
	  f->bcFacet->reservation != size_t(apex.attribute.seeFacet) ||
	  f->caFacet->reservation != size_t(apex.attribute.seeFacet) ) {
	ok = false;
      }
    }
    return ok;
  }

  void resetReservation(vertexT apex, slice<facetT**, facetT**> toDelete) {
    for (auto f: toDelete) {
      f->reservation = -1;
      f->abFacet->reservation = -1;
      f->bcFacet->reservation = -1;
      f->caFacet->reservation = -1;
    }
  }

  bool checkReset() { // todo remove
    bool ok = true;

    auto fVisit = [&](_edge<facetT, vertexT> e) { return true; };

    auto fDo = [&](_edge<facetT, vertexT> e) {
		 if (e.ff->reservation != -1) ok = false;
	       };

    auto fStop = [&](){ return false; };

    context->dfsFacet(context->H, fVisit, fDo, fStop);
    return ok;
  }

  void redistribute(slice<facetT**, facetT**> facetsBeneath,
		    slice<facetT**, facetT**> newFacets) {
    auto canSee = [&](facetT* f, vertexT p) {
		    return visible(f, p) &&
		      f->a != p && f->b != p && f->c != p;//todo
		  };

    // Redistribute the outside points
    if (facetsBeneath.size() == 1) {
      // Only one facet affected, simply distribute among three facets

      facetT* fdel = facetsBeneath[0];
      size_t fn = fdel->size();

#ifdef SEQ_SPLIT
      for (size_t i=0; i<fn; ++i) {
	fdel->at(i).attribute.seeFacet = nullptr;
	for (int j=0; j<3; ++j) {
	  if (canSee(newFacets[j], fdel->at(i))) {
	    fdel->at(i).attribute.seeFacet = newFacets[j];
	    newFacets[j]->push_back(fdel->at(i));
	    break;
	  }
	}
      }
#else
      auto flag = sequence<int>(fn);
      parallel_for(0, fn, [&](size_t i) {
			    flag[i] = 3;
			    fdel->at(i).attribute.seeFacet = nullptr;
			    for (int j=0; j<3; ++j) {
			      if (canSee(newFacets[j], fdel->at(i))) {
				flag[i] = j;
				fdel->at(i).attribute.seeFacet = newFacets[j];
				break;
			      }
			    }
			  });
      // Partition points based on the facets assigned to
      auto tmpBuffer = sequence<vertexT>(fn);
      auto splits = split_k(4, fdel->seeList, make_slice(tmpBuffer), flag);

      size_t fn2 = scan_inplace(make_slice(splits), addm<size_t>());
      splits.push_back(fn2);

      parallel_for(0, fn, [&](size_t i) {fdel->at(i) = tmpBuffer[i];});

      for (int j=0; j<3; ++j)
	newFacets[j]->seeList =
	  fdel->seeList.cut(splits[j], splits[j+1]);
#endif

    } else {
      // Multiple facets affected, new memory needed

      int nf = facetsBeneath.size();
      int nnf = newFacets.size();

      size_t fn = 0;
      for(int j=0; j<nf; ++j) {
	fn += facetsBeneath[j]->size();
      }

#ifdef SEQ_SPLIT
      for(int i=0; i<nf; ++i) { // Old facet loop
	for(size_t j=0; j<facetsBeneath[i]->size(); ++j) { // Point loop
	  facetsBeneath[i]->at(j).attribute.seeFacet = nullptr;
	  for (int k=0; k<nnf; ++k) { // New facet loop
	    if (canSee(newFacets[k], facetsBeneath[i]->at(j))) {
	      facetsBeneath[i]->at(j).attribute.seeFacet = newFacets[k];
	      newFacets[k]->push_back(facetsBeneath[i]->at(j));
	      break;
	    }
	  }
	}
      }
#else
      auto tmpBuffer = sequence<vertexT>(fn);
      fn = 0;
      for(int j=0; j<nf; ++j) {
	parallel_for(0, facetsBeneath[j]->size(),
		     [&](size_t x){tmpBuffer[fn+x] = facetsBeneath[j]->at(x);});
	fn += facetsBeneath[j]->size();
      }

      auto flag = sequence<int>(fn);
      parallel_for(0, fn, [&](size_t i) {
			    flag[i] = nnf;
			    tmpBuffer[i].attribute.seeFacet = nullptr;
			    for (int j=0; j<nnf; ++j) {
			      if (canSee(newFacets[j], tmpBuffer[i])) {
				flag[i] = j;
				tmpBuffer[i].attribute.seeFacet = newFacets[j];
				break;
			      }
			    }
			  });

      // Partition points based on the facets assigned to
      auto tmpBuffer2 = new sequence<vertex3d>(fn);//todo free
      auto splits = split_k(nnf+1, make_slice(tmpBuffer), tmpBuffer2->cut(0, tmpBuffer2->size()), flag);


      size_t fn2 = scan_inplace(make_slice(splits), addm<size_t>());
      splits.push_back(fn2);

      for (int j=0; j<nnf; ++j)
	newFacets[j]->seeList =
	  tmpBuffer2->cut(splits[j], splits[j+1]);
#endif

    }
  }
};

////////////////////////////
// Initialization
////////////////////////////

template<class pt>
conflictList<linkedFacet3d<vertex3d>, vertex3d> *makeInitialHull(slice<pt*, pt*> P) {
  using linkedFacet3d = linkedFacet3d<vertex3d>;

  // Find 8 extrema assuming no duplicate
  auto xx = minmax_element(P, [&](pt i, pt j) {return i[0]<j[0];});
  size_t xMin = xx.first - &P[0];
  size_t xMax = xx.second - &P[0];
  assert(xMin != xMax);

  auto yy = minmax_element(P, [&](pt i, pt j) {return i[1]<j[1];});
  size_t yMin = yy.first - &P[0];
  size_t yMax = yy.second - &P[0];
  assert(yMin != yMax);

  auto zz = minmax_element(P, [&](pt i, pt j) {return i[2]<j[2];});
  size_t zMin = zz.first - &P[0];
  size_t zMax = zz.second - &P[0];
  assert(zMin != zMax);

  // Initialize points with visible facet link
  auto Q = sequence<vertex3d>(P.size());
  parallel_for(0, P.size(), [&](size_t i) {
			      for(int j=0; j<P[i].dim; ++j)
				Q[i][j] = P[i][j];
			    });

  // Make initial facets
  auto f0 = makeFacet(Q[xMin], Q[yMin], Q[zMin]);
  auto f1 = makeFacet(Q[xMax], Q[yMin], Q[zMin]);
  auto f2 = makeFacet(Q[xMin], Q[yMax], Q[zMin]);
  auto f3 = makeFacet(Q[xMax], Q[yMax], Q[zMin]);
  auto f4 = makeFacet(Q[xMin], Q[yMin], Q[zMax]);
  auto f5 = makeFacet(Q[xMax], Q[yMin], Q[zMax]);
  auto f6 = makeFacet(Q[xMin], Q[yMax], Q[zMax]);
  auto f7 = makeFacet(Q[xMax], Q[yMax], Q[zMax]);

  linkFacet(f0, f2, f1, f4);
  linkFacet(f1, f5, f0, f3);
  linkFacet(f2, f6, f3, f0);
  linkFacet(f3, f1, f2, f7);
  linkFacet(f4, f0, f5, f6);
  linkFacet(f5, f7, f4, f1);
  linkFacet(f6, f4, f7, f2);
  linkFacet(f7, f3, f6, f5);

#ifdef SEQ_SPLIT
  for(size_t i=0; i<P.size(); i++) {
    if (visible(f0, Q[i])) {
      Q[i].attribute.seeFacet = f0;
      f0->push_back(Q[i]);
    } else if (visible(f1, Q[i])) {
      Q[i].attribute.seeFacet = f1;
      f1->push_back(Q[i]);
    } else if (visible(f2, Q[i])) {
      Q[i].attribute.seeFacet = f2;
      f2->push_back(Q[i]);
    } else if (visible(f3, Q[i])) {
      Q[i].attribute.seeFacet = f3;
      f3->push_back(Q[i]);
    } else if (visible(f4, Q[i])) {
      Q[i].attribute.seeFacet = f4;
      f4->push_back(Q[i]);
    } else if (visible(f5, Q[i])) {
      Q[i].attribute.seeFacet = f5;
      f5->push_back(Q[i]);
    } else if (visible(f6, Q[i])) {
      Q[i].attribute.seeFacet = f6;
      f6->push_back(Q[i]);
    } else if (visible(f7, Q[i])) {
      Q[i].attribute.seeFacet = f7;
      f7->push_back(Q[i]);
    } else {
      Q[i].attribute.seeFacet = nullptr;
    }
  }
#else

  // Assign each point to one of the visible facets
  auto flag = sequence<int>(P.size());
  parallel_for(0, P.size(), [&](size_t i) {
			      if (visible(f0, Q[i])) {
				flag[i] = 0; Q[i].attribute.seeFacet = f0;
			      } else if (visible(f1, Q[i])) {
				flag[i] = 1; Q[i].attribute.seeFacet = f1;
			      } else if (visible(f2, Q[i])) {
				flag[i] = 2; Q[i].attribute.seeFacet = f2;
			      } else if (visible(f3, Q[i])) {
				flag[i] = 3; Q[i].attribute.seeFacet = f3;
			      } else if (visible(f4, Q[i])) {
				flag[i] = 4; Q[i].attribute.seeFacet = f4;
			      } else if (visible(f5, Q[i])) {
				flag[i] = 5; Q[i].attribute.seeFacet = f5;
			      } else if (visible(f6, Q[i])) {
				flag[i] = 6; Q[i].attribute.seeFacet = f6;
			      } else if (visible(f7, Q[i])) {
				flag[i] = 7; Q[i].attribute.seeFacet = f7;
			      } else {
				flag[i] = 8; Q[i].attribute.seeFacet = nullptr;
			      }
			    });

  // Partition points based on the facets assigned to

  auto Q2 = new sequence<vertex3d>(Q.size()); //todo free

  auto splits = split_k(9, make_slice(Q), make_slice(Q2->begin(), Q2->end()), flag);

  size_t n = scan_inplace(make_slice(splits), addm<size_t>());
  splits.push_back(n);

  // Assign points to the respecive facets
  f0->seeList = Q2->cut(splits[0], splits[1]);
  f1->seeList = Q2->cut(splits[1], splits[2]);
  f2->seeList = Q2->cut(splits[2], splits[3]);
  f3->seeList = Q2->cut(splits[3], splits[4]);
  f4->seeList = Q2->cut(splits[4], splits[5]);
  f5->seeList = Q2->cut(splits[5], splits[6]);
  f6->seeList = Q2->cut(splits[6], splits[7]);
  f7->seeList = Q2->cut(splits[7], splits[8]);
#endif

  auto context = new _hull<linkedFacet3d, vertex3d>(f0);

  return new conflictList(context);
}

////////////////////////////
// Main
////////////////////////////

template<class pt>
sequence<facet3d<pt>> incrementHull3dSerial(slice<pt*, pt*> P) {
  using facet3d = facet3d<pt>;
  using linkedFacet3d = linkedFacet3d<vertex3d>;

#ifdef WRITE
  {
    ofstream myfile;
    myfile.open("hull.txt", std::ofstream::trunc); myfile.close();
    myfile.open("point.txt", std::ofstream::trunc);
    for(size_t p=0; p<P.size(); ++p)
      myfile << P[p] << p << endl;
    myfile.close();
  }
#endif

  timer t; t.start();

  auto cg = makeInitialHull(P);
  _hull<linkedFacet3d, vertex3d> *context = cg->context;

  cout << "init-time = " << t.stop() << endl;

  size_t round = 0;
  double apexTime = 0;
  double frontierTime = 0;
  double createTime = 0;
  double splitTime = 0;

  while (true) {
    t.start();

    // Serial

    //vertex3d apex = cg->randomApex();
    vertex3d apex = cg->furthestApex();

#ifdef VERBOSE
    cout << ">>>>>>>>>" << endl;
    context->printHull();
    cout << "apex chosen = " << apex << endl;
#endif
    apexTime += t.get_next();

    if (apex.isEmpty()) break;

    round ++;

#ifdef WRITE
    context->writeHull();
#endif

    auto frontier = cg->computeFrontier(apex);
    auto frontierEdges = get<0>(frontier);
    auto facetsBeneath = get<1>(frontier);

#ifdef VERBOSE
    cout << "frontier = ";
    for(auto e: *frontierEdges)
      cout << e.a << "," << e.b << " ";
    cout << endl;
    cout << "to delete = ";
    for(auto f: *facetsBeneath)
      cout << *f << " ";
    cout << endl;
#endif
    frontierTime += t.get_next();

    // Create new facets
    auto newFacets = sequence<linkedFacet3d*>(frontierEdges->size());

    for (size_t i=0; i<frontierEdges->size(); ++i) {
      _edge e = frontierEdges->at(i);
      newFacets[i] = makeFacet(e.a, e.b, apex);
    }

#ifdef VERBOSE
    cout << "to create = ";
    for (size_t i=0; i<frontierEdges->size(); ++i)
      cout << *(newFacets[i]) << " ";
    cout << endl;
#endif

    // Connect new facets
    for (size_t i=0; i<frontierEdges->size(); ++i) {
      linkFacet(newFacets[i],
		newFacets[(i+1)%frontierEdges->size()],
		frontierEdges->at(i).ff,
		newFacets[(i-1+frontierEdges->size())%frontierEdges->size()]
		);
    }

    context->H = newFacets[0];

    createTime += t.get_next();

    cg->redistribute(facetsBeneath->cut(0, facetsBeneath->size()), make_slice(newFacets));

    splitTime += t.stop();

    // Delete existing facets
    for(int j=0; j<facetsBeneath->size(); ++j)
      delete facetsBeneath->at(j);

    delete frontierEdges;
    delete facetsBeneath;

  }
  cout << "apex-time = " << apexTime << endl;
  cout << "frontier-time = " << frontierTime << endl;
  cout << "create-time = " << createTime << endl;
  cout << "split-time = " << splitTime << endl;
  cout << "#-rounds = " << round << endl;

#ifdef VERBOSE
  cout << "hull-size = " << context->hullSize() << endl;
#endif

#ifdef WRITE
  context->writeHull();
#endif
  //free stuff

  // todo
  auto dummy = sequence<facet3d>(); dummy.reserve(1);
  return dummy;
}


template<class pt>
sequence<facet3d<pt>> incrementHull3d(slice<pt*, pt*> P) {
  using facet3d = facet3d<pt>;
  using linkedFacet3d = linkedFacet3d<vertex3d>;

#ifdef WRITE
  {
    ofstream myfile;
    myfile.open("hull.txt", std::ofstream::trunc); myfile.close();
    myfile.open("point.txt", std::ofstream::trunc);
    for(size_t p=0; p<P.size(); ++p)
      myfile << P[p] << p << endl;
    myfile.close();
  }
#endif

  timer t; t.start();

  auto cg = makeInitialHull(P);
  _hull<linkedFacet3d, vertex3d> *context = cg->context;

  cout << "init-time = " << t.stop() << endl;

  size_t round = 0;
  double apexTime = 0;
  double frontierTime = 0;
  double splitTime = 0;

  while (true) {
    t.start();

    // Serial

    //vertex3d apex = cg->randomApex();

    //todo if multi-point does not work well, it makes more sense
    // to parallelize this step
    vector<vertex3d> apexes = cg->furthestApexes(8); // todo tune

    apexTime += t.get_next();

    if (apexes.size() <= 0) break;
    round ++;

#ifdef VERBOSE
    cout << ">>>>>>>>>" << endl;
    cout << "apexes = ";
    for (auto x: apexes) cout << x << " "; cout << endl;
#endif

#ifdef WRITE
    context->writeHull();
#endif

    size_t numApex = apexes.size();
    sequence<_edge<linkedFacet3d, vertex3d>>* FE[numApex];
    sequence<linkedFacet3d*>* FB[numApex];

    // todo threshold to run serial algorithm

    // Compute frontier and reserve facets
    parallel_for(0, numApex, [&](size_t a) {
			       auto frontier = cg->computeFrontierAndReserve(apexes[a]);

			       sequence<_edge<linkedFacet3d, vertex3d>>* frontierEdges = get<0>(frontier);
			       sequence<linkedFacet3d*>* facetsBeneath = get<1>(frontier);
			       FE[a] = frontierEdges;
			       FB[a] = facetsBeneath;
			     });

    frontierTime += t.get_next();

    // Check reservation for success, then reset reservation
    bool success[numApex];
    // size_t c = 0;
    parallel_for(0, numApex, [&](size_t a) {
			       if (!cg->confirmReservation( apexes[a], FB[a]->cut(0, FB[a]->size()) )) {
				 success[a] = false;
			       } else {
				 c++;
				 success[a] = true;
			       }
			     });
    parallel_for(0, numApex, [&](size_t a) {
			       cg->resetReservation(apexes[a], FB[a]->cut(0, FB[a]->size()));
			     });

    // cout << "hull size = " << context->hullSize() << endl;
    // cout << "num apex = " << numApex << endl;
    // cout << "success = " << c << endl;
    // if (c == 0) abort();

    // todo filter out successful points and use parfor1

    // Process the successful points
    parallel_for(0, numApex, [&](size_t a) {
			       if (success[a]) {

				 // Create new facets
				 auto newFacets = sequence<linkedFacet3d*>(FE[a]->size());

				 for (size_t i=0; i<FE[a]->size(); ++i) {
				   _edge e = FE[a]->at(i);
				   newFacets[i] = makeFacet(e.a, e.b, apexes[a]);
				 }

				 // Connect new facets
				 for (size_t i=0; i<FE[a]->size(); ++i) {
				   linkFacet(newFacets[i],
					     newFacets[(i+1)%FE[a]->size()],
					     FE[a]->at(i).ff,
					     newFacets[(i-1+FE[a]->size())%FE[a]->size()]
					     );
				 }

				 context->H = newFacets[0]; // todo data race

				 cg->redistribute(FB[a]->cut(0, FB[a]->size()), make_slice(newFacets));
			       }// END if success
			     });

    //todo remove
    // if (!cg->checkReset()) {// checking this is expensive O(nh)
    //   cout << "not reset " << endl; abort();
    // }

    splitTime += t.stop();

    // Clean up
    parallel_for(0, numApex, [&](size_t a) {
			       if (success[a]) {
				 // Delete existing facets
				 for(int j=0; j<FB[a]->size(); ++j)
				   delete FB[a]->at(j);
			       }

			       delete FE[a];
			       delete FB[a];
			     });

  }
  cout << "apex-time = " << apexTime << endl;
  cout << "frontier-time = " << frontierTime << endl;
  cout << "create-split-time = " << splitTime << endl;
  cout << "#-rounds = " << round << endl;

#ifdef VERBOSE
  cout << "hull-size = " << context->hullSize() << endl;
#endif

#ifdef WRITE
  context->writeHull();
#endif
  //free stuff

  // todo
  auto dummy = sequence<facet3d>(); dummy.reserve(1);
  return dummy;
}
