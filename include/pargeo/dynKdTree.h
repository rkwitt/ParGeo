#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

// Simple dynamic kd-tree

namespace dynKdTree {


  template<int dim, typename floatT = double>
  class coordinate {

  protected:

    floatT data[dim];

  public:

    coordinate(floatT* _data) {

      for (int i = 0; i < dim; ++ i) data[i] = _data[i];

    }

    template<typename T>
    coordinate(T& _data) {

      for (int i = 0; i < dim; ++ i) data[i] = _data[i];

    }

    coordinate() { }

    floatT& operator[](int i) {

      return data[i];

    }

  };


  template<int dim> class boundingBox {

  public:

    coordinate<dim> topLeft, lowerRight;

    boundingBox() { };

    template<typename T>
    boundingBox(std::vector<T>& _input, int s = -1, int e = -1) {

      if (s < 0 || e < 0) {
	s = 0;
	e = _input.size();
      }

      if ((e - s) < 2) return;

      topLeft = coordinate<dim>(_input[s]);
      lowerRight = coordinate<dim>(_input[s]);

      for (int j = s; j < e; ++ j) {
	T p = _input[j];
	for (int i = 0; i < dim; ++ i) {
	  topLeft[i] = std::min(p[i], topLeft[i]);
	  lowerRight[i] = std::max(p[i], lowerRight[i]);
	}
      }

    }

    ~boundingBox() {

    }

  };


  template<int dim, typename T> class baseNode {

  protected:

    static const int threshold = 2; // for splitting

    boundingBox<dim> box;

  public:

    virtual bool internal() { return true; }

    baseNode() { }

    virtual ~baseNode() { };

  };


  template<int dim, typename T>
  class dataNode: public baseNode<dim, T> { // leaf node

    std::vector<T> data;

  public:

    bool internal() { return false; }

    dataNode(std::vector<T>& _input, int s = -1, int e = -1) {

      if (s < 0 || e < 0) {
	s = 0;
	e = _input.size();
      }

      baseNode<dim, T>::box = boundingBox<dim>(_input, s, e);

      data = std::vector<T>();

      for (int i = s; i < e; ++ i) {
	data.push_back(_input[i]);
      }

    }

  };


  template<int dim, typename T>
  class splitNode: public baseNode<dim, T> { // internal node

    baseNode<dim, T>* left = nullptr;

    baseNode<dim, T>* right = nullptr;

    int splitDim = -1;

    double split = -1;

  public:

    splitNode(std::vector<T>& _input, int s = -1, int e = -1, int _splitDim = 0):
      splitDim(_splitDim) {

      if (s < 0 || e < 0) {
	s = 0;
	e = _input.size();
      }

      std::cout << "size = " << e - s << "\n";

      baseNode<dim, T>::box = boundingBox<dim>(_input, s, e);

      std::nth_element(_input.begin() + s,
		       _input.begin() + s + (e - s) / 2,
		       _input.begin() + e,
		       [&](T& a, T& b){
			 return a[splitDim] < b[splitDim];
		       });

      split = _input[s + (e - s) / 2][splitDim];

      if (e - s < baseNode<dim, T>::threshold) {

	left = new dataNode<dim, T>(_input, s, s + (e - s) / 2);

	right = new dataNode<dim, T>(_input, s + (e - s) / 2, e);

      } else {

	left = new splitNode(_input, s, s + (e - s) / 2, (splitDim + 1) % dim);

	right = new splitNode(_input, s + (e - s) / 2, e, (splitDim + 1) % dim);

      }

    }

    ~splitNode() {

      delete left;
      delete right;

    }

  };


}; // End namespace dynKdTree
