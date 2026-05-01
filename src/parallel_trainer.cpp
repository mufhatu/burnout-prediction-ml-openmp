#include "parallel_trainer.h"
#include <omp.h>
#include <algorithm>

/**
 * Parallel training function using OpenMP.
 * Implements linear regression with gradient descent and feature normalization.
 */

void ParallelTrainer::train(const std::vector<std::vector<double>>& X,
                             std::vector<double>& y,
                             std::vector<double>& w,
                             double& b) {
 
    if (X.empty()) return; 
                                
    int n = (int)X.size();         //number of samples (rows)
    int f = (int)X[0].size();      //number of features (columns)

    // Step 1: Feature normalization
    // (Min-Max scaling)
    // ==============================
    std::vector<double> xmin(f, 1e18), xmax(f, -1e18);
    for (const auto& row : X)
        for (int j = 0; j < f; j++) {
            xmin[j] = std::min(xmin[j], row[j]);
            xmax[j] = std::max(xmax[j], row[j]);
        }

    // Normalize dataset into [0,1] range    
    std::vector<std::vector<double>> Xn(n, std::vector<double>(f));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < f; j++) {
            double range = xmax[j] - xmin[j];
            Xn[i][j] = (range == 0) ? 0.0 : (X[i][j] - xmin[j]) / range;
        }
    
   
    // Step 2: Initialize model
    // ==============================    
    w.assign(f, 0.0);
    b = 0.0;

    double lr = 0.01;
    int epochs = 1000;
    
    // Step 3: Gradient Descent Loop
    // ==============================
    for (int e = 0; e < epochs; e++) {
        std::vector<double> dw(f, 0.0);
        double db = 0.0;

        
        // Step 4: Parallel computation
        // ==============================
        #pragma omp parallel
        {
             // Thread-local gradients (avoid race conditions)
            std::vector<double> dw_local(f, 0.0);
            double db_local = 0.0;
      
            #pragma omp for nowait
            for (int i = 0; i < n; i++) {

                // Compute prediction: y = w·x + b
                double pred = b;
                for (int j = 0; j < f; j++)
                    pred += w[j] * Xn[i][j];

                 // Compute error    
                double error = pred - y[i];

                // Accumulate gradients for weights
                for (int j = 0; j < f; j++)
                    dw_local[j] += error * Xn[i][j];
                db_local += error;
            }
 
            // Step 5: Merge thread results
            // ==============================
            #pragma omp critical
            { 
                for (int j = 0; j < f; j++)
                    dw[j] += dw_local[j];
                db += db_local;
            }
        }

        // Step 6: Update model parameters
        // ==============================
        for (int j = 0; j < f; j++)
            w[j] -= lr * (dw[j] / n);
        b -= lr * (db / n);
    }
}