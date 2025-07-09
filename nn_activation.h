#pragma once
#include "tensor.h"
#include "nn_layer.h"

namespace utec::neural_network {

    template <typename T>
    class ReLU : public ILayer<T> {
        Tensor<T,2> mask;
    public:
        Tensor<T,2> forward(const Tensor<T,2>& x) override {
            Tensor<T,2> result(x.shape()[0], x.shape()[1]);
            mask = Tensor<T,2>(x.shape()[0], x.shape()[1]);

            for (size_t i = 0; i < x.shape()[0]; ++i) {
                for (size_t j = 0; j < x.shape()[1]; ++j) {
                    mask.at(i, j) = x.at(i, j) > 0 ? 1 : 0;
                    result.at(i, j) = x.at(i, j) * mask.at(i, j);
                }
            }
            return result;
        }

        Tensor<T,2> backward(const Tensor<T,2>& grad) override {
            Tensor<T,2> result(grad.shape()[0], grad.shape()[1]);
            for (size_t i = 0; i < grad.shape()[0]; ++i) {
                for (size_t j = 0; j < grad.shape()[1]; ++j) {
                    result.at(i, j) = grad.at(i, j) * mask.at(i, j);
                }
            }
            return result;
        }
    };

} // namespace utec::neural_network