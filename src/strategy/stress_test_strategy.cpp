#include <iostream>
#include <chrono>
#include "strategy/stress_test_strategy.h"
#include "sequential_trainer.h"
#include "parallel_trainer.h"

/**
 * StressTestStrategy evaluates performance scaling of the ML pipeline.
 * It compares Sequential vs Parallel training across increasing dataset sizes.
 *
 * This is used to measure scalability and speedup.
 */
void StressTestStrategy::run(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const std::vector<double>&,
    double,
    const std::vector<double>&,
    double)
{
    std::cout << "\n=== Stress Test ===\n";
    std::cout << "-----------------------------------------\n";
    std::cout << "Rows\tSeq(ms)\tPar(ms)\tSpeedup\n";
    std::cout << "-----------------------------------------\n";

    // Test scalability by increasing dataset size
    for (int mult : {5, 10, 20, 50}) {

        // Step 1: Expand dataset
        // ==============================
        std::vector<std::vector<double>> X_big;
        std::vector<double> y_big;

        X_big.reserve(X.size() * mult);
        y_big.reserve(y.size() * mult);

        for (int i = 0; i < mult; i++) {
            X_big.insert(X_big.end(), X.begin(), X.end());
            y_big.insert(y_big.end(), y.begin(), y.end());
        }

        // Step 2: Sequential training
        // ==============================
        std::vector<double> w1, w2;
        double b1 = 0, b2 = 0;

        auto t1 = std::chrono::high_resolution_clock::now();
        SequentialTrainer().train(X_big, y_big, w1, b1);
        auto t2 = std::chrono::high_resolution_clock::now();

        // Step 3: Parallel training
        // ==============================
        auto t3 = std::chrono::high_resolution_clock::now();
        ParallelTrainer().train(X_big, y_big, w2, b2);
        auto t4 = std::chrono::high_resolution_clock::now();

        // Step 4: Compute execution times
        // ==============================
        auto seq_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        auto par_time = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

        // Avoid division by zero
        double speedup = (par_time > 0) ? (double)seq_time / par_time : 0;

        // Step 5: Output results
        // ==============================
        std::cout << X_big.size() << "\t"
                  << seq_time << "\t"
                  << par_time << "\t"
                  << speedup << "\n";
    }
}