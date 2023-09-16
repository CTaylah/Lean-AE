#include "NeuralNetwork.h"
#include <iostream>
#include <ctime>
#include <omp.h>
std::mt19937 rng;
void vectorToPPM(Eigen::VectorXi example, const std::string& filename);

int main(int argc, char** argv){

    srand(time(NULL));

    Eigen::MatrixXi examples(784, 60000);
    Eigen::MatrixXi labels(10, 60000);

    Eigen::Ref<Eigen::MatrixXi> examplesRef = examples; 
    Eigen::Ref<Eigen::MatrixXi> labelsRef = labels;
    
    bool success = ReadMNISTImages("MNIST/training_images/train-images.idx3-ubyte", examplesRef);
    bool success2 = ReadMNISTLabels("MNIST/training_labels/train-labels.idx1-ubyte", labelsRef);
    if(!success || !success2){
        std::cout << "Failed to read MNIST data" << std::endl;
        return 1;
    }

    Eigen::MatrixXd examplesDouble = examplesRef.cast<double>();
    Eigen::MatrixXd labelsDouble = labelsRef.cast<double>();
    examplesDouble = examplesDouble / 255.0;

    double cost = 0;

    NeuralNetwork neuralNetwork({784, 256, 256, 784});

    int batch_size = omp_get_num_threads();
    double total_cost = 0.0;
    for (int e = 0; e < 10; ++e) {
        double epoch_cost = 0.0;
        for (int i = 0; i < 300; i += batch_size) {
            Eigen::MatrixXd batch = examplesDouble.block(0, i, 784, batch_size);
            neuralNetwork.BackpropagateBatch(batch, batch, 0.029, epoch_cost);
        }
        std::cout << "Epoch: " << e << " Cost: " << epoch_cost << std::endl;
    }
    
    int index = rand() % 10;
    double meanSquaredError = Math::MeanSquaredError(neuralNetwork.GetPrediction(examplesDouble.col(index)), examplesDouble.col(index));
    std::cout << meanSquaredError << std::endl;
    int answer;
    for(int i = 0; i < 10; ++i){
        if(labelsDouble(i, index) == 1){
            answer = i;
        }
    }

    Eigen::VectorXd prediction = neuralNetwork.GetPrediction(examplesDouble.col(index));
    vectorToPPM((prediction * 255).cast<int>(), "Prediction");
    vectorToPPM((examplesDouble.col(index) * 255).cast<int>(), "Input");

}
void vectorToPPM(Eigen::VectorXi example, const std::string& filename)
{
    std::ofstream file("output/" + filename);
    file << "P3\n28 28\n255\n";
    for (int i = 0; i < example.size(); i++)
    {
        if(example(i) < 0)
            example(i) = 0;
        else if(example(i) > 255)
            example(i) = 255;
        file << example(i)<< " " << example(i) << " " << example(i) << " ";
        if ((i + 1) % 28 == 0)
        {
            file << "\n";
        }
    }
    file.close();
}