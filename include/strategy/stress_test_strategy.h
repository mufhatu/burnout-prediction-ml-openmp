#pragma once
#include "analysis_strategy.h"

/**
 * @brief Concrete strategy that benchmarks sequential vs parallel training.
 *
 * Runs both trainers across four increasing dataset sizes to measure
 * execution time and compute speedup, demonstrating the scalability
 * of the OpenMP parallelised implementation.
 */
class StressTestStrategy : public AnalysisStrategy {
public:
     /**
     * @brief Runs the stress test benchmark across multiple dataset sizes.
     *
     * @param X Feature matrix (rows = samples, cols = features).
     * @param y True label vector (burnout level per sample).
     * @param w_seq Weight vector from the sequential trainer (unused).
     * @param b_seq Bias from the sequential trainer (unused).
     * @param w_par Weight vector from the parallel trainer (unused).
     * @param b_par Bias from the parallel trainer (unused).
     */
    void run(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        const std::vector<double>& w_seq,
        double b_seq,
        const std::vector<double>& w_par,
        double b_par
    ) override;
};