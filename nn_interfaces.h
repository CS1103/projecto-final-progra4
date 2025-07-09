#pragma once
#include "nn_layer.h"

namespace utec::neural_network {
  using namespace algebra;
  template<typename T>
  class ReLU final : public ILayer<T> {
  public:
    Tensor<T,2> forward(const Tensor<T,2>& z) override;
    Tensor<T,2> backward(const Tensor<T,2>& g) override;
  };

  template<typename T>
  class Sigmoid final : public ILayer<T> {
  public:
    Tensor<T,2> forward(const Tensor<T,2>& z) override;
    Tensor<T,2> backward(const Tensor<T,2>& g) override;
  };
}