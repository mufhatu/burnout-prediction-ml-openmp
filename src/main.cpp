#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include "sequential_trainer.h"
#include "parallel_trainer.h"

void loadCSV(const std::string& filename,
             std::vector<std::vector<double>>& X,
             std::vector<double>& y);

const std::vector<std::string> FEATURE_NAMES = {
    "age", "experience_years", "daily_work_hours", "sleep_hours",
    "caffeine_intake", "bugs_per_day", "commits_per_day",
    "meetings_per_day", "screen_time", "exercise_hours", "stress_level"
};

std::string decodeLabel(double value) {
    if (value < 0.5) return "Low";
    else if (value < 1.5) return "Medium";
    else return "High";
}

int main() {
    
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    
    loadCSV("../data/developer_burnout_dataset.csv", X, y);

    // Sequential training
    std::vector<double> w_seq;
    double b_seq = 0;

    auto start = std::chrono::high_resolution_clock::now();

    SequentialTrainer seq;
    seq.train(X, y, w_seq, b_seq);
    auto end = std::chrono::high_resolution_clock::now();
    auto seq_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Sequential Training complete\n";
    for (int j = 0; j < (int)w_seq.size(); j++)
        std::cout << "  " << FEATURE_NAMES[j] << ": " << w_seq[j] << "\n";
    std::cout << "b = " << b_seq << "\n";
    std::cout << "Time: " << seq_time.count() << "ms\n\n";

    // Parallel training
    std::vector<double> w_par;
    double b_par = 0;

    start = std::chrono::high_resolution_clock::now();
    ParallelTrainer par;
    par.train(X, y, w_par, b_par);
    end = std::chrono::high_resolution_clock::now();
    auto par_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Parallel Training complete\n";
    for (int j = 0; j < (int)w_par.size(); j++)
        std::cout << "  " << FEATURE_NAMES[j] << ": " << w_par[j] << "\n";
    std::cout << "b = " << b_par << "\n";
    std::cout << "Time: " << par_time.count() << "ms\n\n";

    // Speedup
    double speedup = (double)seq_time.count() / par_time.count();
    std::cout << "Speedup: " << speedup << "x\n\n";


    // ── Predictions ───────────────────────────────────────────────────
    std::cout << "Sample Predictions (first 5):\n";
    std::cout << "----------------------------------------\n";
    for (int i = 0; i < 5 && i < (int)X.size(); i++) {
        double pred_seq = b_seq;
        double pred_par = b_par;
        for (int j = 0; j < (int)w_seq.size(); j++) {
            pred_seq += w_seq[j] * X[i][j];
            pred_par += w_par[j] * X[i][j];
        }
        std::cout << "Actual: " << decodeLabel(y[i])
                  << " | Seq: "  << decodeLabel(pred_seq)
                  << " | Par: "  << decodeLabel(pred_par) << "\n";
    }
    return 0;
}