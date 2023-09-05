#include "NeuralNetwork.h"

#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <vector>

using Eigen::MatrixXd; 
using Eigen::VectorXd;

void readMNISTImage(std::vector<int>& pixelData);

void readMINSTLabel(std::vector<int>& label);

void printImage(const std::vector<int>& image);



int main()
{
    srand(time(NULL));

    std::vector<int> pixelData;
    readMNISTImage(pixelData);
    std::cout << pixelData.size() << std::endl;

    std::vector<Eigen::VectorXd> images;

    for(int i = 0; i < pixelData.size(); i+=784)
    {
        Eigen::VectorXd imageVector(784);
        for(int j = 0; j < 784; j++)
        {
            imageVector(j) = pixelData[i+j];
        }
        images.push_back(imageVector);
    }

     
    std::vector<int> labels;
    readMINSTLabel(labels);

    std::vector<Eigen::VectorXd> labelVectors;
    for(int i = 0; i < labels.size(); i++)
    {
        Eigen::VectorXd labelVector(10);
        for(int j = 0; j < 10; j++)
        {
            if(labels[i] == j)
            {
                labelVector(j) = 1;
            }
            else
            {
                labelVector(j) = 0;
            }
        }
        labelVectors.push_back(labelVector);
    }


    Layer layer1(784, 16);
    Layer layer2(16, 16);
    Layer layer3(16, 16);
    Layer layer4(16, 10);

    Network network;
    network.push_back(layer1);
    network.push_back(layer2);
    network.push_back(layer3);
    network.push_back(layer4);
    
    network.learn(0.1, 35, 400, labelVectors, images);



    return 0;

}



void readMNISTImage(std::vector<int>& image)
{
    std::ifstream file("MNIST/training_images/train-images.idx3-ubyte", std::ios::binary);

    if(!file){
        std::cout << "Error: file could not be opened" << std::endl;
        return;
    }

    //Read magic number
    int magicNumber = 0;
    file.read((char*)&magicNumber, sizeof(magicNumber));
    magicNumber = __builtin_bswap32(magicNumber);

//    Read number of images
    int numImages = 0;
    file.read((char*)&numImages, sizeof(numImages));
    numImages = __builtin_bswap32(numImages);


    //Read number of rows
    int numRows = 0;
    file.read((char*)&numRows, sizeof(numRows));
    numRows = __builtin_bswap32(numRows);

    //Read number of columns
    int numCols = 0;
    file.read((char*)&numCols, sizeof(numCols));
    numCols = __builtin_bswap32(numCols);

    //Read image data
    for(int i = 0; i < numImages; i++){
        for(int r = 0; r < numRows; r++){
            for(int c = 0; c < numCols; c++){
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                image.push_back((int)temp);
            }
        }
    }
    if(file.fail()){
        std::cerr << "error while reading file" << std::endl;
        return;
    }
}

void readMINSTLabel(std::vector<int>& label){
    std::ifstream file("MNIST/training_labels/train-labels.idx1-ubyte", std::ios::binary);

    if(!file){
        std::cout << "Error: file could not be opened" << std::endl;
        return;
    }

    //Read magic number
    int magicNumber = 0;
    file.read((char*)&magicNumber, sizeof(magicNumber));
    magicNumber = __builtin_bswap32(magicNumber);

    //Read number of labels
    int numLabels = 0;
    file.read((char*)&numLabels, sizeof(numLabels));
    numLabels = __builtin_bswap32(numLabels);

    //Read label data
    for(int i = 0; i < numLabels; i++){
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        label.push_back((int)temp);
    }
    if(file.fail()){
        std::cerr << "error while reading file" << std::endl;
        return;
    }
}

   

