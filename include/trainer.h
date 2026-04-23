#pragma once
#include <vector>

class Trainer {
public :
    virtual void train(std::vector<double>& x,
                       std::vector<double>& y,
                       double& w,
                       double& b) = 0;

    virtual ~Trainer(){}
};