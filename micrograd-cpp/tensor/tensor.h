#pragma once

#include <memory>
#include <vector>
#include <cassert>
#include <span>

// access submatrix/vectors
// print
// use initializer_list instead of vector

template <typename T>
class tensor {
public:
    tensor(std::vector<size_t> shape) : shape(shape) {
        offset = 0;
        size = 1;
        for (int dim : shape) {
            if(dim <= 0)
                throw std::runtime_error("Dimension size must be greater than 0");
            size *= dim;
        }
        data = std::shared_ptr<T[]>(new T[size]);
        strides.resize(shape.size());
        strides.back() = 1;
        for (int i = shape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    };

    T item() {
        if (!(shape.size() == 1 && shape[0] == 1)) {
            throw std::runtime_error("item() only works on tensors with one element.");
        }
        return data[offset];
    }

    static tensor<T> ones(std::vector<size_t> shape) {
        tensor<T> t(shape);
        for (size_t i = 0; i < t.size; i++) {
            t.data[i] = 1;
        }
        return t;
    }

    static tensor<T> constants(std::vector<T> data) {
        std::vector<size_t> shape = {data.size()};
        tensor<T> t(shape);
        std::copy(data.begin(), data.end(), t.data.get());
        return t;
    }

    static tensor<T> constants(std::vector<std::vector<T>> data) {
        if (data.empty()) {
            throw std::runtime_error("Input data cannot be empty.");
        }
        size_t subvector_size = data[0].size();
        for (const auto& subvector : data) {
            if (subvector.size() != subvector_size) {
                throw std::runtime_error("All subvectors must be the same size.");
            }
        }
        std::vector<size_t> shape = {data.size(), subvector_size};
        tensor<T> t(shape);
        T* ptr = t.data.get();
        for (const auto& subvector : data) {
            std::copy(subvector.begin(), subvector.end(), ptr);
            ptr += subvector_size;
        }
        return t;
    }

    void print() {
        if (shape.size() > 2) {
            throw std::runtime_error("print() only works on tensors with one or two dimensions.");
        }
        if (shape.size() == 1) {
            std::cout << "[";
            for (size_t i = 0; i < shape[0]; i++) {
                std::cout << data[i];
                if (i != shape[0] - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]";
        } else if (shape.size() == 2) {
            std::cout << "[\n";
            for (size_t i = 0; i < shape[0]; i++) {
                std::cout << "  [";
                for (size_t j = 0; j < shape[1]; j++) {
                    std::cout << data[i * strides[0] + j * strides[1]];
                    if (j != shape[1] - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]\n";
            }
            std::cout << "]\n";
        }
    }
    

private:
    size_t size;
    size_t offset;
    std::vector<size_t> shape;
    std::shared_ptr<T[]> data;
    std::vector<size_t> strides;
};