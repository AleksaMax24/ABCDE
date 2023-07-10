#include "pch.h"
#include "ABCDE_SSM/summary_statistics/NeuralNetwork.h"
#include "ABCDE_SSM/summary_statistics/kernel_computer.h"
#include "ABCDE_SSM/summary_statistics/summary_statistics_computer.h"

Deep::Deep() {}

Deep::Deep(const string& param)
{
	config_file = param;
	act_with_config_file();
}

Deep& Deep::operator=(const Deep& other)
{
	propTree = other.propTree;
	config_file = other.config_file;
	keys = other.keys;
	index_in_keys = other.index_in_keys;
	config_file = other.config_file;
	deep_exe = other.deep_exe;
	index_score = other.index_score;
	count_snp = other.count_snp;
	index_n = other.index_n;
	index_l = other.index_l;
	index_seed = other.index_seed;
    errorMode = other.errorMode;
    train_neural_number = other.train_neural_number;
	return *this;
}

void Deep::act_with_config_file()
{
    boost::property_tree::ini_parser::read_ini(config_file, propTree);
	deep_exe = propTree.get<std::string>("abcde.name_exe_file");
    index_score = stoi(propTree.get<std::string>("abcde.index_score", "0"));
	count_snp = stoi(propTree.get<std::string>("abcde.count_snp", "0"));
	index_n = stoi(propTree.get<std::string>("abcde.index_n", "0"));
	index_l = stoi(propTree.get<std::string>("abcde.index_l", "0"));
	index_seed = stoi(propTree.get<std::string>("abcde.index_seed", "0"));

    train_neural_number = stoi(propTree.get<std::string>("abcde.train_neural_number", "0"));
	errorMode = get_error_mode("BASE");//propTree.get<std::string>("abcde.error_mode"));

	vector<string> str_list;//, str_index;
	string s = propTree.get<std::string>("abcde.keys");
	boost::split(str_list, s, boost::is_any_of(";"));
	for (int i = 0; i < str_list.size(); i++)
	{
		keys.push_back(str_list[i]);
	}
	s = propTree.get<std::string>("abcde.index_in_keys");
	boost::split(str_list, s, boost::is_any_of(";"));
	for (int i = 0; i < str_list.size(); i++)
	{
		index_in_keys.push_back(stoi(str_list[i]));
	}
}

/**
 * run deepmethod and get error. Remember last ssm data or nullptr if not correct finish.
 * */
double Deep::run(int iter, int element_number, int seed)
{
    ofstream deep_log("log_deep.txt", std::ios::app);

	double res;
	namespace bp = boost::process;
	bp::ipstream is;
	std::vector<std::string> data;
	std::string line;
	cout << bp::search_path(deep_exe).string() + " --default-name=" + tmp_config_file << endl;

    try {
        bp::child c(bp::search_path(deep_exe).string() + " --default-name=" + tmp_config_file , bp::std_out > is);

        while (c.running() && std::getline(is, line) && !line.empty()) {
            deep_log << line + " iteration = " << iter << " element number = " << element_number << " seed = " << seed
                 << endl;
            data.push_back(line);
        }
        c.wait();

        last_output = data.back();
        res = get_error(this->errorMode);

    } catch(std::exception& e) {
        cout << "Exception while getting error, return max value as error." << e.what() << endl;
        res = 1.e+12;
    }
    catch(...) {
        cout << "Exception while getting error, return max value as error." << endl;
        res = 1.e+12;
    }

    try {
//        std::remove(tmp_config_file.c_str());
//        std::remove((tmp_config_file + "-deep-output").c_str());
    }  catch(...) { }
    deep_log << "Error for current method is: " << res << endl;
    return res;
}

/**
 * get error base on error_mode value
 * */
double Deep::get_error(ErrorMode mode) {
    switch (mode) {
        case ErrorMode::BASE:
            return get_BASE_error();
            break;
        case ErrorMode::CNN:
            return get_NN_error(mode);
            break;
        case ErrorMode::DNN:
            return get_NN_error(mode);
            break;
        case ErrorMode::MMD:
            return get_MMD_error();
            break;
        default:
            throw std::invalid_argument("Invalid error mode");
    }
}

/**
 * return score of deepmethod
 * */
double Deep::get_BASE_error() {
    double res = parse_result();
    if(res >= 1.e+12) {
        std::cout << "Error in BASE errorMode compute";
    }
    std::cout << "Base error value: " << res << endl;
    return res;
}

/**
 * return error base on Neural network(DNN or CNN) value that gets from [SSM_DATA]
 * */
double Deep::get_NN_error(ErrorMode mode) {
    SSM_DATA* ssmData = get_SSM_DATA();
    if (ssmData == nullptr)
        return 1.e+12;
    try {
        if (neuralNetworkDistribution == nullptr) {
            return 1.e+12;
        }
        MAE mae = MAE();
        NeuralNetworkDifferenceComputer neuralNetworkDifferenceComputer = NeuralNetworkDifferenceComputer(*neuralNetworkDistribution, mae);

        if (ssmData == NULL) {
            std::cout << "Error with get SSMData, check command";
        }
        double res = neuralNetworkDifferenceComputer.compute(ssmData->ssmData, ssmData->inputData);
        std::cout << "Neural network error value: " << res << endl;
        return res;
    } catch(std::exception& e) {
        cout << "Exception while getting network error, return max value as error." << e.what() << endl;
        return 1.e+12;
    }
    catch(...) {
        cout << "Exception while getting error, return max value as error." << endl;
        return 1.e+12;
    }
}

/**
 * return error base on mmd value that gets from [SSM_DATA]
 * */
double Deep::get_MMD_error() {
    SSM_DATA* ssmData = get_SSM_DATA();
    if (ssmData == nullptr) {
        std::cout << "Error with get SSMData, check command" << endl;
        return 1.e+12;
    }
    GaussianKernel gaussianKernel = GaussianKernel(1.0);
    MMD* mmd = new MMD(gaussianKernel);
    double res = mmd->compute(ssmData->ssmData, ssmData->inputData);
    std::cout << "MMD value: " << res << endl;
    return res;
}

/**
 * call ssm with additional -Z 1 argument to get data
 * */
SSM_DATA* Deep::get_SSM_DATA() {
    namespace bp = boost::process;
    bp::ipstream is;
    string key = "default_model.command";
    string base_command = propTree.get<std::string>(key) + " -Z 1";
    string output_deep_file = tmp_config_file + "-deep-output";
    string ssm_final_command = base_command + " " + output_deep_file;
    std::vector<std::string> data;
    std::string line;

    try {
        bp::child c(ssm_final_command, bp::std_out > is);

        while (c.running() && std::getline(is, line) && !line.empty()) {
            data.push_back(line);
        }
        c.wait();

        std::vector<double> input_data = get_data_from_line(data.back());
        std::vector<double> ssm_data = get_data_from_line(*(data.end() - 2));
        this->last_ssm_data = new SSM_DATA(ssm_data, input_data);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        this->last_ssm_data = nullptr;
    }
    catch(...) {
        cout << "Exception while getting error, return max value as error." << endl;
        this->last_ssm_data = nullptr;
    }

    return this->last_ssm_data;
}

/**
 * get vector of double from file line
 * */
std::vector<double> Deep::get_data_from_line(std::string line) {
    std::vector<double> res;
    std::istringstream iss(line);
    double val;
    while (iss >> val) {
        res.push_back(val);
    }

    return res;
}

/**
 * get resulted score
 * */
double Deep::parse_result()
{
	const char* pattern = ":[-+]?[0-9]*\\.?[0-9]+";
	boost::regex re(pattern);
	int i = 1;
	boost::sregex_iterator it(last_output.begin(), last_output.end(), re);
	boost::sregex_iterator end;
	for (; it != end; ++it)
	{
		if(i == index_score)
		{
			return stod(it->str().erase(0, 1));
		}
		i++;
	}
}

void Deep::prepare_tmp_deep_ini_file(Distribution::Thetha thetha, vector<int>& dtype, int seed)
{
	vector<string> delimeters = {" ", ";", "," };
	string str;
	vector<string> split_str;
	int index = 0;
	int add_int;
	string delimeter;
	for (auto& key : keys)
	{
		str = propTree.get<std::string>(key);
		for (auto& d : delimeters)
		{
			boost::split(split_str, str, boost::is_any_of(d));
			if (split_str.size() > 1)
			{
				delimeter = d;
				break;
			}
		}
		boost::split(split_str, str, boost::is_any_of(delimeter));
		if (dtype[index] == 0)
		{
			add_int = (int)thetha.param[index];
			split_str[index_in_keys[index]] = to_string(add_int);
		}
		else
		    split_str[index_in_keys[index]] = to_string(thetha.param[index]);
		
		index += 1;
		string output;
		for (int i = 0; i < split_str.size(); i++)
		{
			output += split_str[i];
			if(i < split_str.size()-1)
			    output += delimeter;
		}
		propTree.put(key, output);
	}
	//add seed
	string key = "default_model.command";
	str = propTree.get<std::string>(key);
	split_str.clear();
	boost::split(split_str, str, boost::is_any_of(" "));
	split_str[index_seed] = to_string(seed);		
	string output;
	for (int i = 0; i < split_str.size(); i++)
	{
		output += split_str[i];
		if (i < split_str.size() - 1)
			output += ' ';
	}
	propTree.put(key, output);
	//end add seed
	string command = propTree.get<std::string>(key);
	split_str.clear();
	boost::split(split_str, command, boost::is_any_of(" "));
	int n = stoi(split_str[index_n]);
	int l = stoi(split_str[index_l]);
	key = "default_model.partsizes";
	string partsizes = propTree.get<std::string>(key);
	split_str.clear();
	boost::split(split_str, partsizes, boost::is_any_of(";"));
	split_str[0] = to_string(n * l);
	split_str[1] = to_string(n + n * count_snp);
	string output_partsizes;
	for (int i = 0; i < split_str.size(); i++)
	{
		output_partsizes += split_str[i];
		if (i < split_str.size() - 1)
			output_partsizes += ';';
	}
	propTree.put(key, output_partsizes);
	write_ini(tmp_config_file, propTree);
}
	
void Deep::create_tmp_deep_ini_file()
{
	const char* name = tmpnam(NULL);
	tmp_config_file = name;
}
