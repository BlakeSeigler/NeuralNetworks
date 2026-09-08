#include <vector>

class Linear {
    private:
        std::vector<float> out_features;
        std::vector<std::vector<float>> weights;
        std::vector<float> bias;
        int input_feature_size;
        int layer_dim;

    public:

        // initialize weights and bias -- start at 0 because i have no other ideas or needs right now
        Linear(int input_feature_size, int layer_dim) {
            weights = std::vector<std::vector<float>>(layer_dim, std::vector<float>(input_feature_size, 0));
            bias = std::vector<float>(layer_dim, 0);
            this->input_feature_size = input_feature_size;
            this->layer_dim = layer_dim;
        }

        std::vector<float> forward(std::vector<float> in_features) {
            // initialize out at - vector
            out_features = std::vector<float>(layer_dim, 0);

            // Matrix multiply
            for (int i = 0; i < layer_dim; i++) {
                for (int j = 0; j < input_feature_size; j++) {
                    out_features[i] += weights[i][j] * in_features[j];
                }
            }

            // add bias
            for (int i = 0; i < layer_dim; i++) {
                out_features[i] += bias[i];
            }

            return out_features;
        }
};