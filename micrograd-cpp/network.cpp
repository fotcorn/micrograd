#include <iterator>
#include <iostream>

#include "dataset.h"

template <int N>
struct Neuron
{
    float weight[N];
    float bias;
    bool relu;
    float grad[N] = {};
};

#include "nn.h"

template <int INPUT, int OUTPUT>
void forward(const Neuron<INPUT> neurons[OUTPUT], const float input[INPUT], float output[OUTPUT]) {
    for (int i = 0; i < OUTPUT; i++) {
        const Neuron<INPUT>& neuron = neurons[i];
        float res = 0.0f;
        for (int j = 0; j < INPUT; j++) {
            res += neuron.weight[j] * input[j];
        }
        res += neuron.bias;

        if (neuron.relu) {
            res = std::max(0.0f, res);
        }
        output[i] = res;
    }
}

template <int INPUT, int OUTPUT>
void zeroGrad(Neuron<INPUT> neurons[OUTPUT]) {
    for (int i = 0; i < OUTPUT; i++) {
        Neuron<INPUT>& neuron = neurons[i];
        for (int j = 0; j < INPUT; j++) {
            neuron.grad[j] = 0;
        }
    }
}

void backward() {
    // training loop
        // set all gradients to 0
        // loop over all examples
            // evaluate example
            // first gradient is equal to loss
            // recursively update gradients
        // adjust network weights based on gradient
}

float evaluate(float x, float y) {
    float input[] = {x , y};
    float outputL1[std::size(LAYER0)];
    float outputL2[std::size(LAYER1)];
    float outputL3[std::size(LAYER2)];

    zeroGrad<2, std::size(LAYER0)>(LAYER0);
    zeroGrad<16, std::size(LAYER1)>(LAYER1);
    zeroGrad<16, std::size(LAYER2)>(LAYER2);
    
    forward<2, std::size(LAYER0)>(LAYER0, input, outputL1);
    forward<16, std::size(LAYER1)>(LAYER1, outputL1, outputL2);
    forward<16, std::size(LAYER2)>(LAYER2, outputL2, outputL3);
    return outputL3[0];
}

int main()
{
    int correct = 0;
    float loss = 0.0f;
    for (std::size_t i = 0; i < std::size(DATASET_LABELS); i++) {
        float inputX = DATASET_VALUES[i][0];
        float inputY = DATASET_VALUES[i][1];
        float label = DATASET_LABELS[i];

        float output = evaluate(inputX, inputY);

        if (output < 0 == label < 0) {
            correct++;
        }

        // hinge loss
        loss += std::max(0.0f, 1.0f - label * output);
    }

    std::cout << "Accuracy: " << correct << "/" << std::size(DATASET_LABELS) << std::endl;
    std::cout << "Loss: " << loss / static_cast<double>(std::size(DATASET_LABELS)) << std::endl;

    return 0;
}
