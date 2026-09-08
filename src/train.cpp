#include <loss.cpp>
#include <linear.cpp>

linear_layer1 = Linear(3, 2);
linear_layer2 = Linear(2, 1);

std::vector<float> x = {1.0f, 2.0f, 3.0f};
std::vector<float> y = {1.0f, 2.0f, 3.0f};

std::vector<float> y_pred = linear_layer1.forward(x);
y_pred = linear_layer2.forward(y_pred);

float loss = mse_loss(y, y_pred);
std::cout << "Loss: " << loss << std::endl;

// now that i have loss, i can ...