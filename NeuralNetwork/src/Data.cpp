
#include "Data.h"


bool ReadMNISTImages(const std::string& filename, Eigen::Ref<Eigen::MatrixXi>& data){


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
                data(j, i) = (int)temp;
            }
        }
 
        return true;
    }
    else{
        return false;
    }


}


bool ReadMNISTLabels(const std::string& filename, Eigen::Ref<Eigen::MatrixXi>& labels){

    std::ifstream file(filename, std::ios::binary);


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
            for(int j = 0; j < 10; ++j){
                if(j == (int)temp){
                    labels(j, i) = 1;
                }
                else{
                    labels(j, i) = 0;
                }
            }
        }

        return true;
    }
    else{
        return false;
    }
}