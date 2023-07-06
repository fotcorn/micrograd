#include <iterator>
#include <iostream>

template <int N>
struct Neuron
{
    float w[N];
    float b;
    bool relu;

    float forward() {

    }
};

#include "nn.h"

template <int INPUT, int OUTPUT>
void forward(const Neuron<INPUT> neurons[OUTPUT], const float input[INPUT], float output[OUTPUT]) {
    for (int i = 0; i < OUTPUT; i++) {
        const Neuron<INPUT>& neuron = neurons[i];
        float res = 0.0f;
        for (int j = 0; j < INPUT; j++) {
            res += neuron.w[j] * input[j];
        }
        res += neuron.b;

        if (neuron.relu) {
            res = std::max(0.0f, res);
        }
        output[i] = res;
    }
}

int main()
{
    constexpr float x = -2.1357211788694928;
    constexpr float y = -1.6040865570336573;
    constexpr float expectedResult = -1.3584242781835707;

    float input[] = {x , y};
    float outputL1[std::size(LAYER0)];
    float outputL2[std::size(LAYER1)];
    float outputL3[std::size(LAYER2)];
    
    forward<2, std::size(LAYER0)>(LAYER0, input, outputL1);
    forward<16, std::size(LAYER1)>(LAYER1, outputL1, outputL2);
    forward<16, std::size(LAYER2)>(LAYER2, outputL2, outputL3);

    std::cout << outputL3[0] << std::endl;
    std::cout << expectedResult << std::endl;

    return 0;
}
