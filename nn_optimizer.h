#pragma once
#include "tensor.h"
#include <cmath>
#include <unordered_map>

namespace utec::neural_network {

template<typename T>
class IOptimizer {
public:
    virtual ~IOptimizer() = default;
    virtual void update(Tensor<T,2>& param, const Tensor<T,2>& grad) = 0;
    virtual void update(Tensor<T,1>& param, const Tensor<T,1>& grad) = 0;
};

template<typename T>
class SGD : public IOptimizer<T> {
    T learning_rate;
public:
    explicit SGD(T learning_rate = 0.01) : learning_rate(learning_rate) {}

    void update(Tensor<T,2>& param, const Tensor<T,2>& grad) override {
        for (size_t i = 0; i < param.shape()[0]; ++i) {
            for (size_t j = 0; j < param.shape()[1]; ++j) {
                param.at(i, j) -= learning_rate * grad.at(i, j);
            }
        }
    }

    void update(Tensor<T,1>& param, const Tensor<T,1>& grad) override {
        for (size_t i = 0; i < param.shape()[0]; ++i) {
            param.at(i) -= learning_rate * grad.at(i);
        }
    }
};

template<typename T>
class Adam : public IOptimizer<T> {
    T learning_rate;
    T beta1, beta2, epsilon;
    size_t t = 0;

    struct Momentums {
        Tensor<T,2> m;
        Tensor<T,2> v;
    };

    std::unordered_map<Tensor<T,2>*, Momentums> m2_;
    std::unordered_map<Tensor<T,1>*, std::pair<Tensor<T,1>, Tensor<T,1>>> m1_;

public:
    explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
        : learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

    void update(Tensor<T,2>& param, const Tensor<T,2>& grad) override {
        t++;
        auto& [m, v] = m2_[&param];

        //Inicializacion
        if (m.shape().empty()) {
            m = Tensor<T,2>(param.shape()[0], param.shape()[1]);
            v = Tensor<T,2>(param.shape()[0], param.shape()[1]);
            m.fill(0);
            v.fill(0);
        }

        //Actualizacion de moments
        for (size_t i = 0; i < param.shape()[0]; ++i) {
            for (size_t j = 0; j < param.shape()[1]; ++j) {
                m.at(i, j) = beta1 * m.at(i, j) + (1 - beta1) * grad.at(i, j);
                v.at(i, j) = beta2 * v.at(i, j) + (1 - beta2) * grad.at(i, j) * grad.at(i, j);
            }
        }

        //Correccion de bias
        Tensor<T,2> m_hat(param.shape()[0], param.shape()[1]);
        Tensor<T,2> v_hat(param.shape()[0], param.shape()[1]);

        T beta1_t = std::pow(beta1, t);
        T beta2_t = std::pow(beta2, t);

        for (size_t i = 0; i < param.shape()[0]; ++i) {
            for (size_t j = 0; j < param.shape()[1]; ++j) {
                m_hat.at(i, j) = m.at(i, j) / (1 - beta1_t);
                v_hat.at(i, j) = v.at(i, j) / (1 - beta2_t);
                param.at(i, j) -= learning_rate * m_hat.at(i, j) / (std::sqrt(v_hat.at(i, j)) + epsilon);
            }
        }
    }

    void update(Tensor<T,1>& param, const Tensor<T,1>& grad) override {
        t++;
        auto& [m, v] = m1_[&param];

        if (m.shape().empty()) {
            m = Tensor<T,1>(param.shape()[0]);
            v = Tensor<T,1>(param.shape()[0]);
            m.fill(0);
            v.fill(0);
        }

        //Actualizar moments
        for (size_t i = 0; i < param.shape()[0]; ++i) {
            m.at(i) = beta1 * m.at(i) + (1 - beta1) * grad.at(i);
            v.at(i) = beta2 * v.at(i) + (1 - beta2) * grad.at(i) * grad.at(i);
        }

        //Correccion de bias
        Tensor<T,1> m_hat(param.shape()[0]);
        Tensor<T,1> v_hat(param.shape()[0]);

        T beta1_t = std::pow(beta1, t);
        T beta2_t = std::pow(beta2, t);

        for (size_t i = 0; i < param.shape()[0]; ++i) {
            m_hat.at(i) = m.at(i) / (1 - beta1_t);
            v_hat.at(i) = v.at(i) / (1 - beta2_t);
            param.at(i) -= learning_rate * m_hat.at(i) / (std::sqrt(v_hat.at(i)) + epsilon);
        }
    }
};

} // namespace utec::neural_network