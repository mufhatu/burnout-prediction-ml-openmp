#include "sequential_trainer.h"
#include <vector>

void SequentialTrainer::train(std::vector<double>& x,
                              std::vector<double>& y,
                              double& w,
                              double& b){

    double lr = 0.0001;
    int epochs = 800;
    
    for (int e= 0; e< epochs; e++){
        double dw = 0;
        double db = 0;

        for (size_t i = 0; i < x.size(); i++){

            double pred= w*x[i] + b;
            double error = pred - y[i];

            dw += error * x[i];
            db += error;
        }

        w -= lr * dw;
        b -= lr * db;
    }
}
