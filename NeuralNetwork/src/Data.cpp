
#include "Data.h"
#include <string>

bool readMNISTImage(Eigen::VectorXd& data, char* filename){


    std::ifstream file(filename, std::ios::binary);

    if(file.is_open()){
        int magicNumber = 0;
        int numImages = 0;
        int numRows = 0;
        int numCols = 0;

        //MNIST uses big-endian, so we need to reverse the bytes with __builtin_bswap32
        file.read((char*)&magicNumber, sizeof(magicNumber));
        magicNumber = __builtin_bswap32(magicNumber);
        file.read((char*)&numImages, sizeof(numImages));
        numImages =  __builtin_bswap32(numImages);
        file.read((char*)&numRows, sizeof(numRows));
        numRows = __builtin_bswap32(numRows);
        file.read((char*)&numCols, sizeof(numCols));
        numCols = __builtin_bswap32(numCols);

        int imageSize = numRows * numCols;

        for(int i = 0; i < numImages; ++i){
            for(int j = 0; j < imageSize; ++j){
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                data(i * imageSize + j) = (double)temp;
            }
        }
        return true;
    }
    else{
        return false;
    }


}

bool readMNISTLabel(Eigen::VectorXd& labels, char* filename){
    std::ifstream file("data/train-labels.idx1-ubyte", std::ios::binary);

    if(file.is_open()){
        int magicNumber = 0;
        int numLabels = 0;

        //MNIST uses big-endian, so we need to reverse the bytes with __builtin_bswap32
        file.read((char*)&magicNumber, sizeof(magicNumber));
        magicNumber = __builtin_bswap32(magicNumber);
        file.read((char*)&numLabels, sizeof(numLabels));
        numLabels =  __builtin_bswap32(numLabels);

        for(int i = 0; i < numLabels; ++i){
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            labels(i) = (double)temp;
        }
        return true;
    }
    else{
        return false;
    }
}