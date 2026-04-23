#pragma once
#include "trainer.h"

class SequentialTrainer : public Trainer {
public :
    void train(std::vector<double>& x,
               std::vector<double>& y,
               double& w,
               double& b) override;

};