#include "CrossValidation.h"

#include <omp.h>


double Testing::Test(NeuralNetwork network, const Eigen::MatrixXd& testingSet, bool verbose){
    Eigen::VectorXd prediction;
    Eigen::VectorXd target;
    double cost = 0.0;

    for(int i = 0; i < testingSet.cols(); i++){
        prediction = network.GetPrediction(testingSet.col(i));
        cost += Math::MeanSquaredError(prediction, testingSet.col(i));
    }

    cost /= testingSet.cols();
    if(verbose){
    std::cout << "Test cost: " << cost << std::endl;

    int index = 0;
    Eigen::VectorXd predictionExample = network.GetPrediction(testingSet.col(index));
    std::cout << "Prediction Cost: " << Math::MeanSquaredError(predictionExample, testingSet.col(index)) << std::endl;
    
    VectorToPPM((predictionExample * 255).cast<int>(), "Prediction");
    VectorToPPM((testingSet.col(index) * 255).cast<int>(), "Input");
    }
    return cost;
}

double Testing::MonteCarloCV(NeuralNetwork network, TrainingSettings settings, Eigen::MatrixXd& examples, double trainingPercent)
{
    if(trainingPercent < 0 || trainingPercent > 1)
        throw std::invalid_argument("MonteCarloCV: trainingPercent must be between 0 and 1");

    Eigen::Ref<Eigen::MatrixXd> examplesRef = examples;
    Math::ShuffleMatrix(examplesRef);

    int trainingSize = examples.cols() * trainingPercent;
    int testSize = examples.cols() - trainingSize;

    Eigen::MatrixXd trainingSet = examples.block(0, 0, examples.rows(), trainingSize);
    Eigen::MatrixXd testingSet = examples.block(0, trainingSize, examples.rows(), testSize);

    network.Train(settings, trainingSet, trainingSet, true);
    double testCost = Test(network, testingSet, true);
    return testCost;

}

//Return true if network1 is had a lower cost than network2
bool Testing::Compare(NeuralNetwork network1, TrainingSettings settings1, NeuralNetwork network2, TrainingSettings settings2, 
    Eigen::MatrixXd& dataSet, double trainingPercent)
{

    double network1TestCost = 0.0;
    double network2TestCost = 0.0;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            network1TestCost = MonteCarloCV(network1, settings1, dataSet, trainingPercent);
        }
        #pragma omp section
        {
            network2TestCost = MonteCarloCV(network2, settings2, dataSet, trainingPercent);
        }
    }
    return network1TestCost < network2TestCost;
}
//Rename MonteCarloCV
std::vector<double> Testing::Compare(std::vector<NeuralNetwork> networks, TrainingSettings settings, 
    Eigen::MatrixXd& dataSet, double trainingPercent)
{
    std::vector<double> costs;
    for(size_t network = 0; network < networks.size(); network++)
    {
        costs.push_back(0.0);
    }
    
    //#pragma omp parallel for 
    for(size_t network = 0; network < networks.size(); network++)
    {
        int threadID = omp_get_thread_num();
        double cost = MonteCarloCV(networks[threadID], settings, dataSet, trainingPercent);
        costs[threadID] = cost; 
    }

    
    return costs;

}


