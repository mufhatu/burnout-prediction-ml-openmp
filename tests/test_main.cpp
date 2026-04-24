#include <gtest/gtest.h>
#include <vector>

#include "encoder.h"
#include "sequential_trainer.h"

//
// TEST 1: Encoder
//
TEST(EncoderTest, LabelEncoding) {
    EXPECT_EQ(encodeLabel("Low"), 0);
    EXPECT_EQ(encodeLabel("Medium"), 1);
    EXPECT_EQ(encodeLabel("High"), 2);
}

//
// TEST 2: Trainer
//
TEST(TrainerTest, ModelLearnsSomething) {

    std::vector<double> x = {10, 20, 30};
    std::vector<double> y = {0, 1, 2};

    double w = 0;
    double b = 0;

    SequentialTrainer trainer;
    trainer.train(x, y, w, b);

    EXPECT_NE(w, 0);   // model updated
}

//
// REQUIRED MAIN FOR GTEST
//
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}