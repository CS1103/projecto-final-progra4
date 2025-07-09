#pragma once
#include "tensor.h"

namespace utec::neural_network {

template <typename T>
class MSELoss {
    Tensor<T,2> last_pred;
    Tensor<T,2> last_target;

public:
    T forward(const Tensor<T,2>& pred, const Tensor<T,2>& target) {
        last_pred = pred;
        last_target = target;
        T loss = 0;

        for (size_t i = 0; i < pred.shape()[0]; ++i) {
            for (size_t j = 0; j < pred.shape()[1]; ++j) {
                T diff = pred.at(i,j) - target.at(i,j);
                loss += diff * diff;
            }
        }
        return loss / (pred.shape()[0] * pred.shape()[1]);
    }

    Tensor<T,2> backward() {
        Tensor<T,2> grad(last_pred.shape()[0], last_pred.shape()[1]);
        T factor = 2.0 / (last_pred.shape()[0] * last_pred.shape()[1]);

        for (size_t i = 0; i < grad.shape()[0]; ++i) {
            for (size_t j = 0; j < grad.shape()[1]; ++j) {
                grad.at(i,j) = factor * (last_pred.at(i,j) - last_target.at(i,j));
            }
        }
        return grad;
    }
};

template<typename T>
class BCELoss {
    Tensor<T,2> last_pred;
    Tensor<T,2> last_target;

public:
    T forward(const Tensor<T,2>& pred, const Tensor<T,2>& target) {
        last_pred = pred;
        last_target = target;
        T loss = 0;

        for (size_t i = 0; i < pred.shape()[0]; ++i) {
            for (size_t j = 0; j < pred.shape()[1]; ++j) {
                loss += -target.at(i,j) * log(pred.at(i,j)) -
                       (1-target.at(i,j)) * log(1-pred.at(i,j));
            }
        }
        return loss / (pred.shape()[0] * pred.shape()[1]);
    }

    Tensor<T,2> backward() {
        Tensor<T,2> grad(last_pred.shape()[0], last_pred.shape()[1]);
        T factor = 1.0 / (last_pred.shape()[0] * last_pred.shape()[1]);

        for (size_t i = 0; i < grad.shape()[0]; ++i) {
            for (size_t j = 0; j < grad.shape()[1]; ++j) {
                grad.at(i,j) = factor * (last_pred.at(i,j) - last_target.at(i,j)) /
                              (last_pred.at(i,j) * (1 - last_pred.at(i,j)));
            }
        }
        return grad;
    }
};

} // namespace utec::neural_network