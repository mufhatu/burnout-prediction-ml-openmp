#pragma once
#include <vector>

/**
 * @brief Abstract interface for burnout analysis strategies.
 *
 * Defines the Strategy pattern — concrete strategies implement run()
 * to perform different types of analysis (prediction, stress test, etc.)
 * without modifying the calling code in main.cpp.
 */
class AnalysisStrategy {
public:
    /**
     * @brief Runs the analysis using trained model weights.
     *
     * @param X Feature matrix (rows = samples, cols = features).
     * @param y True label vector (burnout level per sample).
     * @param w_seq Weight vector from the sequential trainer.
     * @param b_seq Bias from the sequential trainer.
     * @param w_par Weight vector from the parallel trainer.
     * @param b_par Bias from the parallel trainer.
     */
    virtual void run(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        const std::vector<double>& w_seq,
        double b_seq,
        const std::vector<double>& w_par,
        double b_par
    ) = 0;

    virtual ~AnalysisStrategy() = default;
};