#pragma once

#include "ABCDE_SSM/summary_statistics/NeuralNetwork.h"
#include "pch.h"

class SSM_DATA {
public:
    std::vector<double> ssmData;
    std::vector<double> inputData;

    SSM_DATA(const std::vector<double>& ssmData, const std::vector<double>& inputData) {
        this->ssmData = ssmData;
        this->inputData = inputData;
    }
};

enum class ErrorMode {
    BASE,
    CNN,
    DNN,
    MMD
};

/**
 * class that provides to run Deep method.
 *
 * */
class Deep :public Model
{
private:
	pt::ptree propTree;
    double get_BASE_error();
    double get_NN_error(ErrorMode mode);
    double get_MMD_error();
    std::vector<double> parse_result_deep_res();
    std::vector<double> get_data_from_line(std::string line);
    double get_error(ErrorMode mode);
public:
	Deep();
	Deep(const string& param);
	Deep& operator=(const Deep&);
	void act_with_config_file();
	double run(int iter, int element_number, int seed);
    double parse_result();
	void prepare_tmp_deep_ini_file(Distribution::Thetha thetha, vector<int>& dtype, int seed);
	void create_tmp_deep_ini_file();
    SSM_DATA* get_SSM_DATA();

    ErrorMode get_error_mode(const std::string& name) {
        if (name == "BASE") {
            return ErrorMode::BASE;
        } else if (name == "CNN") {
            return ErrorMode::CNN;
        } else if (name == "DNN") {
            return ErrorMode::DNN;
        } else if (name == "MMD") {
            return ErrorMode::MMD;
        } else {
            throw std::invalid_argument("Invalid error mode name");
        }
    }

    string last_output;
	string config_file;
	string tmp_config_file;
	string deep_exe;
	double error;
	vector<string> keys;
	vector<int> index_in_keys;
	int count_snp;
	int index_n, index_l, index_score, index_seed;
    ErrorMode errorMode;
    SSM_DATA* last_ssm_data;
    int train_neural_number;
    NeuralNetworkDistribution* neuralNetworkDistribution = nullptr;
};

