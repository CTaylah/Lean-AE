
#include "Data.h"

#include "Eigen/Dense"

#include "gtest/gtest.h"

#include <filesystem>

TEST(Data, loadMNIST)
{
    Eigen::MatrixXi data(28 * 28, 60000);
    Eigen::MatrixXi labels(10, 60000);

    Eigen::Ref<Eigen::MatrixXi> dataRef = data;
    Eigen::Ref<Eigen::MatrixXi> labelsRef = labels;

    bool success = ReadMNISTImages("MNIST/training_images/train-images.idx3-ubyte", dataRef);
    bool success2 = ReadMNISTLabels("MNIST/training_labels/train-labels.idx1-ubyte", labelsRef);

    EXPECT_TRUE(success);
    EXPECT_TRUE(success2);

    EXPECT_EQ(data.size(), 60000 * 28 * 28);
    EXPECT_EQ(labels.size(), 600000);


}