#include "sequential_trainer.h"
#include <algorithm>

void SequentialTrainer::train(const std::vector<std::vector<double>>& X,
                               std::vector<double>& y,
                               std::vector<double>& w,
                               double& b) {

    if (X.empty()) return;

    int n = (int)X.size();
    int f = (int)X[0].size();

    // Normalize each feature to [0,1]
    std::vector<double> xmin(f, 1e18), xmax(f, -1e18);
    for (const auto& row : X)
        for (int j = 0; j < f; j++) {
            xmin[j] = std::min(xmin[j], row[j]);
            xmax[j] = std::max(xmax[j], row[j]);
        }

    std::vector<std::vector<double>> Xn(n, std::vector<double>(f));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < f; j++) {
            double range = xmax[j] - xmin[j];
            Xn[i][j] = (range == 0) ? 0.0 : (X[i][j] - xmin[j]) / range;
        }

    w.assign(f, 0.0);
    b = 0.0;

    double lr = 0.01;
    int epochs = 1000;

    for (int e = 0; e < epochs; e++) {
        std::vector<double> dw(f, 0.0);  // one gradient per feature
        double db = 0.0;

        for (int i = 0; i < n; i++) {
            double pred = b;
            for (int j = 0; j < f; j++)
                pred += w[j] * Xn[i][j];

            double error = pred - y[i];

            for (int j = 0; j < f; j++)
                dw[j] += error * Xn[i][j];
            db += error;
        }

        for (int j = 0; j < f; j++)
            w[j] -= lr * (dw[j] / n);
        b -= lr * (db / n);
    }
}