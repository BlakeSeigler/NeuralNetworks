float mse_loss(std::vector<float> y_true, std::vector<float> y_pred) {
    float loss = 0;
    for (int i = 0; i < y_true.size(); i++) {
        loss += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    }
    return loss / y_true.size();
}