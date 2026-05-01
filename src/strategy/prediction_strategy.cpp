#include <iostream>
#include "strategy/prediction_strategy.h"

/**
 * Converts a numeric prediction into a categorical label.
 * This is used to interpret model output (regression → classification style output).
 */
std::string PredictionStrategy::decodeLabel(double value) {
    if (value < 0.5) return "Low";
    else if (value < 1.5) return "Medium";
    else return "High";
}

/**
 * Runs prediction analysis comparing Sequential vs Parallel trained models.
 *
 * Parameters:
 * - X: feature dataset
 * - y: actual labels
 * - w_seq, b_seq: weights and bias from sequential model
 * - w_par, b_par: weights and bias from parallel model
 */
void PredictionStrategy::run(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const std::vector<double>& w_seq,
    double b_seq,
    const std::vector<double>& w_par,
    double b_par)
{
    std::cout << "\nSample Predictions:\n";
    std::cout << "-----------------------------------------\n";
    std::cout << "Actual\t\tSeq\t\tPar\n";
    std::cout << "-----------------------------------------\n";

    // Loop over first 5 samples for quick comparison
    for (int i = 0; i < 5 && i < (int)X.size(); i++) {
        
        // Initialize predictions for both models
        double pred_seq = b_seq;
        double pred_par = b_par;

        // Compute linear regression prediction: w·x + b
        for (int j = 0; j < (int)w_seq.size(); j++) {
            pred_seq += w_seq[j] * X[i][j];
            pred_par += w_par[j] * X[i][j];
        }

        // Print actual vs predicted values (converted to labels)
        std::cout << decodeLabel(y[i]) << "\t\t"
                  << decodeLabel(pred_seq) << "\t\t"
                  << decodeLabel(pred_par) << "\n";
    }
}