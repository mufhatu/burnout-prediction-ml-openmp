#include "parallel_trainer.h"
#include <omp.h>
#include <algorithm>

void ParallelTrainer::train(const std::vector<std::vector<double>>& X,
                             std::vector<double>& y,
                             std::vector<double>& w,
                             double& b) {
 
    if (X.empty()) return; 
                                
    int n = (int)X.size();
    int f = (int)X[0].size();

    std::vector<double> xmin(f, 1e18), xmax(f, -1e18);
    for (const auto& row : X)
        for (int j = 0; j < f; j++) {
            xmin[j] = std::min(xmin[j], row[j]);
            xmax[j] = std::max(xmax[j], row[j]);
        }

    std::vector<std::vector<double>> Xn(n, std::vector<double>(f));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < f; j++) {
            double range = xmax[j] - xmin[j];
            Xn[i][j] = (range == 0) ? 0.0 : (X[i][j] - xmin[j]) / range;
        }

    w.assign(f, 0.0);
    b = 0.0;

    double lr = 0.01;
    int epochs = 1000;

    for (int e = 0; e < epochs; e++) {
        std::vector<double> dw(f, 0.0);
        double db = 0.0;

        #pragma omp parallel
        {
            std::vector<double> dw_local(f, 0.0);
            double db_local = 0.0;

            #pragma omp for nowait
            for (int i = 0; i < n; i++) {
                double pred = b;
                for (int j = 0; j < f; j++)
                    pred += w[j] * Xn[i][j];

                double error = pred - y[i];
                for (int j = 0; j < f; j++)
                    dw_local[j] += error * Xn[i][j];
                db_local += error;
            }

            #pragma omp critical
            {
                for (int j = 0; j < f; j++)
                    dw[j] += dw_local[j];
                db += db_local;
            }
        }

        for (int j = 0; j < f; j++)
            w[j] -= lr * (dw[j] / n);
        b -= lr * (db / n);
    }
}