#pragma once
#include <vector>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <random>
#include <type_traits>

namespace utec::algebra {

template <typename T, size_t Rank>
class Tensor {
private:
    std::vector<size_t> shape_;
    std::vector<T> data_;

public:
    // Constructor vacío
    Tensor() = default;

    // Constructor con dimensiones
    explicit Tensor(const std::vector<size_t>& shape) : shape_(shape) {
        if (shape.size() != Rank) {
            throw std::invalid_argument("Shape rank doesn't match tensor rank");
        }
        data_.resize(calculate_total_size());
    }

    // Constructor para tensores 2D
    Tensor(size_t dim1, size_t dim2) {
        static_assert(Rank == 2, "This constructor is only for 2D tensors");
        shape_ = {dim1, dim2};
        data_.resize(dim1 * dim2);
    }

    // Constructor para tensores 1D
    explicit Tensor(size_t dim1) {
        static_assert(Rank == 1, "This constructor is only for 1D tensors");
        shape_ = {dim1};
        data_.resize(dim1);
    }

    // Métodos para manipulación de datos
    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    void fill_random(T min, T max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dis(min, max);
            for (auto& elem : data_) {
                elem = dis(gen);
            }
        } else {
            std::uniform_real_distribution<T> dis(min, max);
            for (auto& elem : data_) {
                elem = dis(gen);
            }
        }
    }

    // Acceso a elementos para tensores 1D
    T& at(size_t i) {
        static_assert(Rank == 1, "This method is only for 1D tensors");
        return data_.at(i);
    }

    const T& at(size_t i) const {
        static_assert(Rank == 1, "This method is only for 1D tensors");
        return data_.at(i);
    }

    // Acceso a elementos para tensores 2D
    T& at(size_t i, size_t j) {
        static_assert(Rank == 2, "This method is only for 2D tensors");
        return data_.at(i * shape_[1] + j);
    }

    const T& at(size_t i, size_t j) const {
        static_assert(Rank == 2, "This method is only for 2D tensors");
        return data_.at(i * shape_[1] + j);
    }

    // Operaciones matemáticas
    Tensor& operator*=(const T& scalar) {
        for (auto& elem : data_) {
            elem *= scalar;
        }
        return *this;
    }

    Tensor operator*(const T& scalar) const {
        Tensor result(*this);
        return result *= scalar;
    }

    // Información de dimensiones
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return data_.size(); }

    // Método slice para dividir el tensor
    Tensor slice(size_t start_row, size_t end_row) const {
        static_assert(Rank == 2, "Slice is only implemented for 2D tensors");
        size_t rows = end_row - start_row;
        size_t cols = shape_[1];

        Tensor result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.at(i, j) = this->at(start_row + i, j);
            }
        }
        return result;
    }

private:
    size_t calculate_total_size() const {
        return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
    }
};

} // namespace utec::algebra