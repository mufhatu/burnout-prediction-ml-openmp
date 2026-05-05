#pragma once
#include <vector>

/**
 * @brief Trains a linear regression model using parallelised gradient descent.
 *
 * Uses OpenMP to parallelise the gradient accumulation across training samples,
 * significantly reducing training time on multi-core systems.
 */
class ParallelTrainer {
public:
     /**
     * @brief Trains the model and updates weights and bias in-place.
     *
     * @param X Feature matrix (rows = samples, cols = features).
     * @param y Label vector (burnout level per sample).
     * @param w Output weight vector, updated after training.
     * @param b Output bias, updated after training.
     */
    void train(const std::vector<std::vector<double>>& X,
               std::vector<double>& y,
               std::vector<double>& w,
               double& b);
};