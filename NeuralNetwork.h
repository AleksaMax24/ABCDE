#ifndef MU_TMP_SCRIPTS_NEURALNETWORK_H
#define MU_TMP_SCRIPTS_NEURALNETWORK_H

#include <vector>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

class NeuralNetworkDistribution {
public:
    NeuralNetworkDistribution() = default;

    explicit NeuralNetworkDistribution(const std::string& filename) {
        net.load(filename);
    }

    void train(const std::vector<std::vector<double>>& train_data, const std::vector<std::vector<double>>& train_labels) {
        std::vector<tiny_dnn::vec_t> notNanTrainData;
        std::vector<tiny_dnn::vec_t> notNanTrainLabels;

        tiny_dnn::adam optimizer; // there can be problems need to be tested before running (problem is nan predicted)
        for (int i = 0; i < train_data.size(); ++i) {
            bool flag = false;
            for (double j : train_data[i]) {
                if (std::isnan(j))
                    flag = true;
            }
            if (!flag) {
                notNanTrainData.emplace_back(train_data[i].begin(), train_data[i].end());
                notNanTrainLabels.emplace_back(train_labels[i].begin(), train_labels[i].end());
            }
        }

        this->net.train<tiny_dnn::mse, tiny_dnn::adam>(optimizer, notNanTrainData, notNanTrainLabels, 512, 5);
    }

    std::vector<double> predict(const  std::vector<double>& value) {
        tiny_dnn::vec_t res = net.predict(tiny_dnn::vec_t (value.begin(), value.end()));
        std::vector<double> std_vec(res.begin(), res.end());

        return std_vec;
    }
    void save(const std::string& filename = "neural_network"){
        net.save(filename);
    }

protected:
    tiny_dnn::network<tiny_dnn::sequential> net;
};

class DNN: public NeuralNetworkDistribution{
public:
    DNN(int data_size, int params_num) {
        net << tiny_dnn::fully_connected_layer(data_size, data_size)
            << tiny_dnn::relu_layer()
            << tiny_dnn::fully_connected_layer(data_size, data_size)
            << tiny_dnn::relu_layer()
            << tiny_dnn::fully_connected_layer(data_size, data_size / 2)
            << tiny_dnn::relu_layer()
            << tiny_dnn::fully_connected_layer(data_size / 2, data_size / 2)
            << tiny_dnn::relu_layer()
            << tiny_dnn::fully_connected_layer(data_size / 2, params_num);
    }
};

class CNN: public NeuralNetworkDistribution{
public:
    CNN(int data_size, int params_num) {
        int firstDataSize = data_size / 2;
        int secondDataSize = data_size / 4;

        net << conv(1, data_size, 3, 1, 1, padding::same)
            << tanh_layer()
            << conv(1, data_size, 3, 1, 1, padding::same)
            << tanh_layer()

            << max_pooling_layer(data_size, 1, 1, data_size - firstDataSize + 1)

            << conv(1, firstDataSize, 3, 1, 1, padding::same)
            << tanh_layer()
            << conv(1, firstDataSize, 3, 1, 1, padding::same)
            << tanh_layer()

            << max_pooling_layer(firstDataSize, 1, 1, firstDataSize - secondDataSize + 1)

            << conv(1, secondDataSize, 3, 1, 1, padding::same)
            << tanh_layer()
            << conv(1, secondDataSize, 3, 1, 1, padding::same)
            << tanh_layer()

            << fully_connected_layer(secondDataSize, secondDataSize)
            << batch_normalization_layer(secondDataSize, 1)
            << tanh_layer()

            << fully_connected_layer(secondDataSize, secondDataSize)
            << batch_normalization_layer(secondDataSize, 1)
            << tanh_layer()

            << fully_connected_layer(secondDataSize, params_num);
    }
};

#endif //MU_TMP_SCRIPTS_NEURALNETWORK_H
