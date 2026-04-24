#include <gtest/gtest.h>
#include <vector>
#include "encoder.h"
#include "sequential_trainer.h"
#include "parallel_trainer.h"

TEST(EncoderTest, LabelEncoding) {
    EXPECT_EQ(encodeLabel("Low"),    0);
    EXPECT_EQ(encodeLabel("Medium"), 1);
    EXPECT_EQ(encodeLabel("High"),   2);
}

TEST(SequentialTrainerTest, ModelLearnsSomething) {
    std::vector<std::vector<double>> X = {{10,1},{20,2},{30,3},{40,4}};
    std::vector<double> y = {0, 1, 2, 1};
    std::vector<double> w;
    double b = 0;

    SequentialTrainer trainer;
    trainer.train(X, y, w, b);

    EXPECT_EQ((int)w.size(), 2);
    EXPECT_NE(b, 0);
}

TEST(ParallelTrainerTest, MatchesSequential) {
    std::vector<std::vector<double>> X = {{10,1},{20,2},{30,3},{40,4}};
    std::vector<double> y = {0, 1, 2, 1};
    std::vector<double> w_seq, w_par;
    double b_seq = 0, b_par = 0;

    SequentialTrainer seq;
    seq.train(X, y, w_seq, b_seq);

    ParallelTrainer par;
    par.train(X, y, w_par, b_par);

    ASSERT_EQ(w_seq.size(), w_par.size());
    for (size_t i = 0; i < w_seq.size(); i++)
        EXPECT_NEAR(w_seq[i], w_par[i], 0.1);  // allow small floating point diff
    EXPECT_NEAR(b_seq, b_par, 0.1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}