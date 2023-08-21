#pragma once

#include <memory>
#include <vector>
#include <cassert>
#include <span>

// access single element
// access submatrix/vectors

template <typename T>
class tensor {
public:
    tensor(std::vector<int> shape) : shape(shape) {
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

    static tensor<T> ones(std::vector<int> shape) {
        tensor<T> t(shape);
        for (uint64_t i = 0; i < t.size; i++) {
            t.data[i] = 1;
        }
        return t;
    }

    static tensor<T> constants(std::vector<T> data) {
        std::vector<int> shape = {static_cast<int>(data.size())};
        tensor<T> t(shape);
        std::copy(data.begin(), data.end(), t.data.get());
        return t;
    }

private:
    size_t size;
    size_t offset;
    std::vector<int> shape;
    std::shared_ptr<T[]> data;
    std::vector<size_t> strides;
};