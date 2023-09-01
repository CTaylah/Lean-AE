#include <iostream>
#include <Eigen/Dense>
 
using Eigen::MatrixXd; 
using Eigen::VectorXd;

int main()
{
    Eigen::Matrix<int, 3,3> m {
        {1,2,5},
        {4,3,7},
        {3,3,6}

    };

    Eigen::Matrix<int, 3, 1> v(1, 2, 5);
    // The [] operator only works for vectors!. Matrix[i,j] yields Matrix[j] !!
    std::cout << m.rows() << std::endl;
    std::cout << v[2] << std::endl;
}