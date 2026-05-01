#include <iostream>
#include "strategy/prediction_strategy.h"

std::string PredictionStrategy::decodeLabel(double value) {
    if (value < 0.5) return "Low";
    else if (value < 1.5) return "Medium";
    else return "High";
}

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

    for (int i = 0; i < 5 && i < (int)X.size(); i++) {
        double pred_seq = b_seq;
        double pred_par = b_par;

        for (int j = 0; j < (int)w_seq.size(); j++) {
            pred_seq += w_seq[j] * X[i][j];
            pred_par += w_par[j] * X[i][j];
        }

        std::cout << decodeLabel(y[i]) << "\t\t"
                  << decodeLabel(pred_seq) << "\t\t"
                  << decodeLabel(pred_par) << "\n";
    }
}