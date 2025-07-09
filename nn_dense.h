#pragma once
#include "tensor.h"
#include "nn_layer.h"

namespace utec::neural_network {

template <typename T>
class Dense : public ILayer<T> {



public:
    Tensor<T,2> W, dW;
    Tensor<T,1> b, db;
    Tensor<T,2> last_x;
    Dense(size_t in_feats, size_t out_feats) {
        W = Tensor<T,2>(in_feats, out_feats);
        dW = Tensor<T,2>(in_feats, out_feats);
        b = Tensor<T,1>(out_feats);
        db = Tensor<T,1>(out_feats);

        //Inicialización
        T limit = sqrt(6.0 / (in_feats + out_feats));
        W.fill_random(-limit, limit);
        b.fill(0.0);
    }

    Tensor<T,2> forward(const Tensor<T,2>& x) override {
        last_x = x;
        Tensor<T,2> output(x.shape()[0], W.shape()[1]);

        for (size_t i = 0; i < x.shape()[0]; ++i) {
            for (size_t j = 0; j < W.shape()[1]; ++j) {
                output.at(i, j) = b.at(j);
                for (size_t k = 0; k < W.shape()[0]; ++k) {
                    output.at(i, j) += x.at(i, k) * W.at(k, j);
                }
            }
        }
        return output;
    }

    Tensor<T,2> backward(const Tensor<T,2>& grad) override {
        for (size_t i = 0; i < W.shape()[0]; ++i) {
            for (size_t j = 0; j < W.shape()[1]; ++j) {
                dW.at(i, j) = 0;
                for (size_t k = 0; k < grad.shape()[0]; ++k) {
                    dW.at(i, j) += last_x.at(k, i) * grad.at(k, j);
                }
            }
        }

        for (size_t j = 0; j < b.shape()[0]; ++j) {
            db.at(j) = 0;
            for (size_t k = 0; k < grad.shape()[0]; ++k) {
                db.at(j) += grad.at(k, j);
            }
        }

        //Gradiente respecto a la entrada
        Tensor<T,2> input_grad(last_x.shape()[0], last_x.shape()[1]);
        for (size_t i = 0; i < input_grad.shape()[0]; ++i) {
            for (size_t j = 0; j < input_grad.shape()[1]; ++j) {
                input_grad.at(i, j) = 0;
                for (size_t k = 0; k < W.shape()[1]; ++k) {
                    input_grad.at(i, j) += grad.at(i, k) * W.at(j, k);
                }
            }
        }

        return input_grad;
    }
};

} // namespace utec::neural_network