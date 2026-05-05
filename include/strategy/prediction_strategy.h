#pragma once
#include "analysis_strategy.h"
#include <string>

/**
 * @brief Concrete strategy that displays sample burnout predictions.
 *
 * Compares predictions from the sequential and parallel trainers
 * against the true labels for the first 5 samples in the dataset.
 */
class PredictionStrategy : public AnalysisStrategy {
public:
    /**
     * @brief Runs sample predictions using both trained models.
     *
     * @param X Feature matrix (rows = samples, cols = features).
     * @param y True label vector (burnout level per sample).
     * @param w_seq Weight vector from the sequential trainer.
     * @param b_seq Bias from the sequential trainer.
     * @param w_par Weight vector from the parallel trainer.
     * @param b_par Bias from the parallel trainer.
     */
    void run(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        const std::vector<double>& w_seq,
        double b_seq,
        const std::vector<double>& w_par,
        double b_par
    ) override;

private:
    /**
     * @brief Converts a numeric prediction to a burnout label string.
     *
     * @param value Numeric prediction value.
     * @return "Low", "Medium", or "High".
     */
    std::string decodeLabel(double value);
};