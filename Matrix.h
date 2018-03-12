#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <memory>

class Matrix{
public:

    Matrix(){
        _width = -1;
        _height = -1;
        _matrix = std::make_shared<std::vector<std::vector<float>>>(0, std::vector<float>(0));
    }
    Matrix(int width, int height){
        _width = width;
        _height = height;
        _matrix = std::make_shared<std::vector<std::vector<float>>>(height, std::vector<float>(width));
        for (int i = 0; i < height; i++){
            for (int j = 0; j < width; j++){
                (*_matrix)[i][j] = (((float)rand() * 2) / (float)RAND_MAX)-1;
            }
        }
    }
    Matrix(std::vector<std::vector<float>> &matrix){
        _width = matrix[0].size();
        _height = matrix.size();
        _matrix = std::make_shared<std::vector<std::vector<float>>>(matrix);
    }
    Matrix(std::vector<float> &matrix){
        _width = matrix.size();
        _height = 1;
        _matrix = std::make_shared<std::vector<std::vector<float>>>(1, matrix);

    }

    ~Matrix(){

    }

    int width() const {return _width;}
    int height() const {return _height;}
    std::vector<std::vector<float>> data() const {return *_matrix;}

    static std::shared_ptr<Matrix> dot(const Matrix &m1, const Matrix &m2){
        try{
            if (m1.width() == m2.height()){
                std::vector<std::vector<float>> out = std::vector<std::vector<float>>(m1.height(), std::vector<float>(m2.width()));
                for (int i = 0; i < m1.height(); i++){
                    for (int j = 0; j < m2.width(); j++){
                        float tempsum = 0;
                        for (int k = 0; k < m1.width(); k++){
                            tempsum += m1.data()[i][k] * m2.data()[k][j];
                        }
                    out[i][j] = tempsum;
                    }
                }
                return  std::make_shared<Matrix>(out);
            }
            else{
                std::string exc = "dotMissmatchMatrixSizeError";
                throw exc;
            }
        }
        catch (std::string e){
            std::cout << e << std::endl;
        }
        return 0;
    }
    std::shared_ptr<Matrix> dot(const Matrix &m1){ return dot(*this, m1);}
    std::shared_ptr<Matrix> dot(std::shared_ptr<Matrix> m1) { return dot(*this, *m1);}
    static std::shared_ptr<Matrix> dot(std::shared_ptr<Matrix> m1, std::shared_ptr<Matrix> m2) {return dot(*m1, *m2);}

    static float sum(Matrix &m){
        float sum = 0;
        for (int i = 0; i < m.height(); i++){
            for (int j = 0; j < m.width(); j++){
                sum += m.data()[i][j];
            }
        }
        return sum;
    }
    float sum(){ return sum(*this);}

    static std::shared_ptr<Matrix> t(Matrix m){
        std::vector<std::vector<float>> out = std::vector<std::vector<float>>(m.width(), std::vector<float>(m.height()));
        for (int i = 0; i < m.height(); i++){
            for (int j = 0; j < m.width(); j++){
                out[j][i] = m.data()[i][j];
            }
        }
        return std::make_shared<Matrix>(out);
    }
    std::shared_ptr<Matrix> t(){
        return t(*this);
    }

    static std::shared_ptr<Matrix> add(Matrix &m1, Matrix &m2){
        try{
            if (m1.width() == m2.width() && m1.height() == m2.height()){
                std::vector<std::vector<float>> out = std::vector<std::vector<float>>(m1.height(), std::vector<float>(m1.width()));
                for (int i = 0; i < m1.height(); i++){
                    for (int j = 0; j < m1.width(); j++){
                        out[i][j] = m1.data()[i][j] + m2.data()[i][j];
                    }
                }
                return  std::make_shared<Matrix>(out);
            }
            else{
                std::string exc = "addMissmatchMatrixSizeError "+m1.size()+","+m2.size();
                throw exc;
            }
        }
        catch (std::string e){
            std::cout << e << std::endl;
        }
        return nullptr;
    }
    std::shared_ptr<Matrix> add(Matrix &m){return add(*this, m);}
    std::shared_ptr<Matrix> add(std::shared_ptr<Matrix> m){return add(*this, *m);}
    static std::shared_ptr<Matrix> add(std::shared_ptr<Matrix> m1, std::shared_ptr<Matrix> m2){return add(*m1, *m2);}

    static std::shared_ptr<Matrix> add(Matrix &m1, float f){
        std::vector<std::vector<float>> out = std::vector<std::vector<float>>(m1.height(), std::vector<float>(m1.width()));
        for (int i = 0; i < m1.height(); i++){
            for (int j = 0; j < m1.width(); j++){
                out[i][j] = m1.data()[i][j] + f;
            }
        }
        return  std::make_shared<Matrix>(out);
    }
    std::shared_ptr<Matrix> add(float f){return add(*this, f);}

    static std::shared_ptr<Matrix> multiply(Matrix &m1, Matrix &m2){
        try{
            if (m1.width() == m2.width() && m1.height() == m2.height()){
                std::vector<std::vector<float>> out = std::vector<std::vector<float>>(m1.height(), std::vector<float>(m1.width()));
                for (int i = 0; i < m1.height(); i++){
                    for (int j = 0; j < m1.width(); j++){
                        out[i][j] = m1.data()[i][j] * m2.data()[i][j];
                    }
                }
                return  std::make_shared<Matrix>(out);
            }
            else{
                std::string exc = "multMissmatchMatrixSizeError";
                throw exc;
            }
        }
        catch (std::string e){
            std::cout << e << std::endl;
        }
        return nullptr;
    }
    std::shared_ptr<Matrix> multiply(Matrix &m){return multiply(*this, m);}
    std::shared_ptr<Matrix> multiply(std::shared_ptr<Matrix> m){return multiply(*this, *m);}
    static std::shared_ptr<Matrix> multiply(std::shared_ptr<Matrix> m1, std::shared_ptr<Matrix> m2){return multiply(*m1, *m2);}

    static std::shared_ptr<Matrix> multiply(Matrix &m1, float f){
        std::vector<std::vector<float>> out = std::vector<std::vector<float>>(m1.height(), std::vector<float>(m1.width()));
        for (int i = 0; i < m1.height(); i++){
            for (int j = 0; j < m1.width(); j++){
                out[i][j] = m1.data()[i][j] * f;
            }
        }
        return  std::make_shared<Matrix>(out);
    }
    std::shared_ptr<Matrix> multiply(float f){return multiply(*this, f);}
    static std::shared_ptr<Matrix> multiply(std::shared_ptr<Matrix> m, float f){return multiply(*m, f);}

    std::string size(){
        std::string out = "";
        return out + "(" + std::to_string(_width) +","+ std::to_string(_height) +")";
    }

    friend std::ostream& operator<<(std::ostream& os, Matrix& m){
        for (int i = 0; i < m.height()-1; i++){
            for (int j = 0; j < m.width(); j++){
                os << m.data()[i][j] << " ";
            }
            os << std::endl;
        }
        for (int j = 0; j < m.width(); j++){
            os << m.data()[m.height()-1][j] << " ";
        }
        return os;
    }

    std::string to_string(){
        std::string out = "";
        for (int i = 0; i < this->height()-1; i++){
            for (int j = 0; j < this->width(); j++){
                out += std::to_string(this->data()[i][j]) + " ";
            }
            out += "\n";
        }
        for (int j = 0; j < this->width(); j++){
            out += std::to_string(this->data()[this->height()-1][j]) + " ";
        }
        return out;
    }

private:
    std::shared_ptr<std::vector<std::vector<float>>> _matrix;
    int _width = 0;
    int _height = 0;
};
