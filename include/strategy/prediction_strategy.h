#pragma once
#include "analysis_strategy.h"
#include <string>

class PredictionStrategy : public AnalysisStrategy {
public:
    void run(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        const std::vector<double>& w_seq,
        double b_seq,
        const std::vector<double>& w_par,
        double b_par
    ) override;

private:
    std::string decodeLabel(double value);
};