#pragma once
#include <vector>

class AnalysisStrategy {
public:
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