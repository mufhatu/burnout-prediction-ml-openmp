#pragma once
#include <vector>

class SequentialTrainer {
public:
    void train(const std::vector<std::vector<double>>& X,
               std::vector<double>& y,
               std::vector<double>& w,
               double& b);
};