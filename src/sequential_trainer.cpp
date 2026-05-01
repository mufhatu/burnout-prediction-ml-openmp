#include "sequential_trainer.h"
#include <algorithm>

/**
 * SequentialTrainer
 * ------------------
 * Trains a linear regression model using gradient descent (CPU single-threaded version).
 * Also performs feature normalization (min-max scaling).
 */
void SequentialTrainer::train(const std::vector<std::vector<double>>& X,
                               std::vector<double>& y,
                               std::vector<double>& w,
                               double& b) {
    
    // Safety check: empty dataset
    if (X.empty()) return;

    int n = (int)X.size();      // number of samples
    int f = (int)X[0].size();   // number of features

    // Normalize each feature to [0,1]
    // ==============================
    std::vector<double> xmin(f, 1e18), xmax(f, -1e18);

    // Find min and max for each feature
    for (const auto& row : X)
        for (int j = 0; j < f; j++) {
            xmin[j] = std::min(xmin[j], row[j]);
            xmax[j] = std::max(xmax[j], row[j]);
        }

    // Normalize dataset
    std::vector<std::vector<double>> Xn(n, std::vector<double>(f));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < f; j++) {
            double range = xmax[j] - xmin[j];
            Xn[i][j] = (range == 0) ? 0.0 : (X[i][j] - xmin[j]) / range;
        }

    // Step 2: Initialize parameters
    // ==============================

    w.assign(f, 0.0);     // weights initialized to zero
    b = 0.0;              // bias initialized to zero

    double lr = 0.01;     // learning rate
    int epochs = 1000;    // training iterations

    // Step 3: Gradient Descent Loop
    // ==============================

    for (int e = 0; e < epochs; e++) {
        std::vector<double> dw(f, 0.0);  // one gradient per feature
        double db = 0.0;                 // gradient for bias

        // Step 4: Compute gradients
        // ==============================

        for (int i = 0; i < n; i++) {

            // Linear prediction: y = w·x + b
            double pred = b;
            for (int j = 0; j < f; j++)
                pred += w[j] * Xn[i][j];

            // Compute error    
            double error = pred - y[i];

            // Accumulate gradients
            for (int j = 0; j < f; j++)
                dw[j] += error * Xn[i][j];
            db += error;
        }

        // Step 5: Update parameters
        // ==============================
        for (int j = 0; j < f; j++)
            w[j] -= lr * (dw[j] / n);
        b -= lr * (db / n);
    }
}