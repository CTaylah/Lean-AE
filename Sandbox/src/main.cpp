#include "NeuralNetwork.h"

#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <vector>

using Eigen::MatrixXd; 
using Eigen::VectorXd;

void readMNISTImage(std::vector<int>& image);

void readMINSTLabel(int& label);

void printImage(const std::vector<int>& image);



int main()
{
    srand(time(NULL));
    std::vector<int> image;
    readMNISTImage(image);
    
    int imageLabel = 0;
    readMINSTLabel(imageLabel);

    Eigen::VectorXd imageVector(image.size());

    imageVector = imageVector / 255.0;

    for(int i = 0; i < image.size(); i++)
    {
        imageVector(i) = image[i];
    }

    Network network(imageVector);
    Layer layer1(16, 784);
    Layer layer2(16, 16);
    Layer layer3(10, 16);
    network.push_back(layer1);
    network.push_back(layer2);
    network.push_back(layer3);

    auto output = network.feedForward();

    std::cout << output << std::endl;



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

    //Read number of images
    int numImages = 0;
    file.read((char*)&numImages, sizeof(numImages));
    numImages = __builtin_bswap32(numImages);

    numImages = 1;

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

void readMINSTLabel(int& label){
    std::ifstream file("MNIST/training_labels/train-labels.idx1-ubyte", std::ios::binary);

    if(!file){
        std::cout << "Error: file could not be opened" << std::endl;
        return;
    }

    //Read magic number
    int magicNumber = 0;
    file.read((char*)&magicNumber, sizeof(magicNumber));
    magicNumber = __builtin_bswap32(magicNumber);

    //Read number of images
    int numImages = 0;
    file.read((char*)&numImages, sizeof(numImages));
    numImages = __builtin_bswap32(numImages);

    numImages = 1;

    //Read label data
    for(int i = 0; i < numImages; i++){
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        label = (int)temp;
    }
    if(file.fail()){
        std::cerr << "error while reading file" << std::endl;
        return;
    }
}

void printImage(const std::vector<int>& image){
    for(int i = 0; i < image.size(); i++){
        std::cout << image[i] << "  ";
        if(i % 28 == 0){
            std::cout << std::endl;
        }
    }
}
