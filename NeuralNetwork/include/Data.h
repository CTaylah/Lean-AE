#pragma once

#include "Eigen/Dense"

#include <iostream>
#include <string>
#include <fstream>



[[nodiscard]] bool ReadMNISTImages(const std::string& filename, Eigen::Ref<Eigen::MatrixXi>& data);

[[nodiscard]] bool ReadMNISTLabels(const std::string& filename, Eigen::Ref<Eigen::MatrixXi>& labels);

void VectorToPPM(Eigen::VectorXi example, const std::string& filename);
