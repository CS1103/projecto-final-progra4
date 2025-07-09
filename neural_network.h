#pragma once
#include "nn_layer.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include <vector>
#include <memory>

#include "nn_dense.h"

namespace utec::neural_network {
    using namespace algebra;
template <typename T>
class NeuralNetwork {
    std::vector<std::unique_ptr<ILayer<T>>> layers;
    MSELoss<T> criterion;


public:
    std::unique_ptr<IOptimizer<T>> optimizer;
    NeuralNetwork() = default;
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers.push_back(std::move(layer));
    }

    void set_optimizer(std::unique_ptr<IOptimizer<T>> opt) {
        optimizer = std::move(opt);
    }

    Tensor<T,2> forward(const Tensor<T,2>& x) {
        Tensor<T,2> output = x;
        for (auto& layer : layers) {
            output = layer->forward(output);
        }
        return output;
    }

    void backward(const Tensor<T,2>& grad) {
        Tensor<T,2> current_grad = grad;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            current_grad = (*it)->backward(current_grad);
        }
    }

    void optimize() {
        for (auto& layer : layers) {
            if (auto dense = dynamic_cast<Dense<T>*>(layer.get())) {
                optimizer->update(dense->W, dense->dW);
                optimizer->update(dense->b, dense->db);
            }
        }
    }

    void train(const Tensor<T,2>& X, const Tensor<T,2>& Y, size_t epochs, size_t batch_size = 32) {
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            T total_loss = 0;
            size_t num_batches = (X.shape()[0] + batch_size - 1) / batch_size;

            for (size_t batch = 0; batch < num_batches; ++batch) {
                size_t start = batch * batch_size;
                size_t end = std::min(start + batch_size, X.shape()[0]);
                Tensor<T,2> x_batch = X.slice(start, end);
               Tensor<T,2> y_batch = Y.slice(start, end);

                // Forward pass
                Tensor<T,2> output = forward(x_batch);
                T loss = criterion.forward(output, y_batch);
                total_loss += loss;

                // Backward pass
                Tensor<T,2> grad = criterion.backward();
                backward(grad);

                // Update parameters
                optimize();
            }

            std::cout << "Epoch " << epoch + 1 << "/" << epochs
                      << ", Loss: " << total_loss / num_batches << std::endl;
        }
    }
};
}
