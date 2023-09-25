#include "NeuralNetwork.h"
#include <omp.h>


//Note, this is only setup for an autoencoder right now
void KFold(int k, NeuralNetwork network, const Eigen::MatrixXd& examples)
{
    //First, split the examples into k folds
    std::vector<Eigen::MatrixXd> folds(k);
    int foldSize = examples.cols() / k;
    for(int i = 0; i < k; i++)
    {
        folds[i] = examples.block(0, i * foldSize, examples.rows(), foldSize);
    }
    TrainingSettings settings(0.0115, 10, omp_get_num_threads());

    //Now, train the network k times, each time using a different fold as the validation set
    for(int i = 0; i < k; i++)
    {
        
    }


}