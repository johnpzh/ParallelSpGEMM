//
// Created by Peng, Zhen on 4/18/23.
//

#include "../include/test.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int i, int j) {
  return i + j;
}

void test() {

}

PYBIND11_MODULE(add_test, m) {
//PYBIND11_MODULE(add_test, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring

  m.def("add", &add, "A function that adds two numbers");
}