#include "pch.h"

Abcde::Abcde() {}

Abcde::Abcde(const string& param)
{
	config_file = param;
	act_with_config_file();
	init_posterior();
	error = new double[count_iter];
}

void Abcde::normalize_weights()
{
	double sum = 0.0;
	for (int i = 0; i < count_iter; i++)
	{
		sum += new_posterior.w[i];
	}
	for (int i = 0; i < count_iter; i++)
	{
		new_posterior.w[i] = new_posterior.w[i] / sum;
	}
}

void Abcde::set_sample_dist_param()//now bounds param
{
	double sum;
	for (int i = 0; i < count_opt_param; i++)
	{
		sum = 0.0;
		for (int j = 0; j < count_iter; j++)
		{
			sum += set_bounds(posterior.thetha[j].param[i], lbound[i], hbound[i]);
		}
		sample_mean[i] = sum / count_iter;
		sum = 0.0;
		for (int j = 0; j < count_iter; j++)
		{
			sum += (set_bounds(posterior.thetha[j].param[i], lbound[i], hbound[i]) - sample_mean[i]) * (set_bounds(posterior.thetha[j].param[i], lbound[i], hbound[i]) - sample_mean[i]);
		}
		sample_std[i] = sqrt(sum / (count_iter - 1));
	}
	sum = 0.0;
	for (int j = 0; j < count_iter; j++)
	{
		sum += posterior.error[j];
	}
	sample_error_mean = sum / count_iter;
}

double Abcde::max_weight(double* w)
{
	double max_w = -1.0;
	for (int i = 0; i < count_iter; i++)
	{
		if (w[i] > max_w)
			max_w = w[i];
	}
	return max_w;
}

Distribution::Thetha Abcde::get_prev_iter_with_weight()
{
	/*	double prop;
		for (int i = 0; i < count_iter; i++)
		{
			prop = generator.prior_distribution(Distribution::TYPE_DISTR::RANDOM, 0.0, max_weight(posterior.w));
			if (posterior.w[i] >= prop) {
				return posterior.thetha[i];
			}
		}
		return posterior.thetha[0];*/

	double prop = generator.prior_distribution(Distribution::TYPE_DISTR::RANDOM, 0.0, 1.0);
	double sum_prop = 0.0;
	for (int i = 0; i < count_iter; i++)
	{
		if (prop > sum_prop && prop < sum_prop + posterior.w[i])
			return posterior.thetha[i];
		sum_prop += posterior.w[i];
	}
	return posterior.thetha[best_index];
}

int Abcde::get_index_best()
{
	double min_error = 10;
	best_index = 0;
	for (int i = 0; i < count_iter; i++)
	{
		if (new_posterior.error[i] <= min_error)
		{
			best_index = i;
			min_error = new_posterior.error[i];
		}
	}
	return best_index;
}

double Abcde::set_new_weight(const int curr_index)
{
	ofstream logfile("log_weight.txt", std::ios::app);
	double phi = 1.0, sum = 0.0, norm, _x, _mean, _std, res;
	if (print_add_log)
	{
		logfile << "curr_index = " << curr_index << endl;
		logfile << "best_index = " << best_index << endl;
	}
	for (int i = 0; i < count_opt_param; i++)
	{
		_x = set_bounds(curr_thetha.param[i], lbound[i], hbound[i]);
		_mean = set_bounds(new_posterior.thetha[best_index].param[i], lbound[i], hbound[i]);
		_std = abs(sample_mean[i] - set_bounds(new_posterior.thetha[best_index].param[i], lbound[i], hbound[i]));
		if (print_add_log)
		{
			logfile << "i = " << i << endl;
			logfile << "x = " << _x << endl;
			logfile << "mean = " << _mean << endl;
			logfile << "std = " << _std << endl;
		}
		_std = min(max(_std, 1.0 / count_opt_param), abs(hbound[i] - lbound[i]) / count_opt_param);
		if (print_add_log) logfile << "final std = " << _std << endl;
		phi *= generator.kernel_function(Distribution::TYPE_DISTR::NORM_WITH_PARAM, _x, _mean, _std);
		if (print_add_log) logfile << "phi = " << phi << endl;

	}
	_x = new_posterior.error[curr_index];
	_mean = new_posterior.error[best_index];
	_std = abs(new_posterior.error[best_index] - sample_error_mean);
	if (print_add_log)
	{
		logfile << "error:" << endl;
		logfile << "x = " << _x << endl;
		logfile << "mean = " << _mean << endl;
		logfile << "std = " << _std << endl;
	}
	_std = max(_std, 0.0001);//////////////////
	if (print_add_log) logfile << "final std = " << _std << endl;
	phi *= generator.kernel_function(Distribution::TYPE_DISTR::NORM_WITH_PARAM, _x, _mean, _std);
	if (print_add_log) logfile << "phi = " << phi << endl;
	for (int i = 0; i < count_iter; i++)
	{
		if (print_add_log) logfile << "i = " << i << endl;
		norm = 1.0;
		for (int j = 0; j < count_opt_param; j++)
		{
			if (print_add_log) 		logfile << "j = " << j << endl;
			_x = set_bounds(posterior.thetha[i].param[j], lbound[j], hbound[j]);
			_mean = set_bounds(curr_thetha.param[j], lbound[j], hbound[j]);
			if (print_add_log)
			{
				logfile << "x = " << _x << endl;
				logfile << "mean = " << _mean << endl;
			}
			_std = 2.0 * sample_std[j];
			if (print_add_log) logfile << "std = " << _std << endl;
			_std = min(max(_std, 1.0 / count_opt_param), abs(hbound[j] - lbound[j]) / count_opt_param);
			if (print_add_log) logfile << "final std = " << _std << endl;
			norm *= generator.kernel_function(Distribution::TYPE_DISTR::NORM_WITH_PARAM, _x, _mean, _std);//new_posterior or posterior???
			if (print_add_log) 		logfile << "norm = " << norm << endl;
		}
		if (print_add_log) logfile << "posterior.w[" << i << "] = " << posterior.w[i] << endl;
		sum += posterior.w[i] * norm;
		if (print_add_log) logfile << "sum = " << sum << endl;
	}
	res = phi / sum;
	if (print_add_log) logfile << "res = " << res << endl;
	res = max(res, 1.0 / count_opt_param);
	if (print_add_log) logfile << "final res = " << res << endl;
	logfile.close();
	return res;
}

Distribution::Thetha Abcde::generate_vector_param(Distribution::TYPE_DISTR mode)
{
	Distribution::Thetha thetha;
	for (int i = 0; i < count_opt_param; i++)
	{
		thetha.param.push_back(generator.prior_distribution(mode, mean[i], std[i]));
	}
	return thetha;
}

void Abcde::update_posterior()
{
	for(int i = 0; i < count_iter; i++)
	{
		curr_thetha = new_posterior.thetha[i];
		new_posterior.w[i] = set_new_weight(i);
	}
	normalize_weights();
}

void  Abcde::act_with_config_file()
{
	boost::property_tree::ptree pt;
	boost::property_tree::ini_parser::read_ini(config_file, pt);
	count_thread = stoi(pt.get<std::string>("abcde.count_thread", "0"));
	deep_exe = pt.get<std::string>("abcde.name_exe_file", "0");
	t = stoi(pt.get<std::string>("abcde.t", "0"));
	count_iter = stoi(pt.get<std::string>("abcde.count_iter", "0"));
	start_iter = stoi(pt.get<std::string>("abcde.start_iter", "0"));
	count_opt_param = stoi(pt.get<std::string>("abcde.count_opt_param", "0"));
	bounds_crossing_mode = stoi(pt.get<std::string>("abcde.bounds_crossing_mode", "0"));
	crossing_mode = stoi(pt.get<std::string>("abcde.crossing_mode", "0"));
	print_add_log = stoi(pt.get<std::string>("abcde.print_add_log", "0"));
	vector<string> str_mean, str_std, str_hbound, str_lbound, str_dtype, str_cross_dist_mean, str_cross_dist_std, str_mut_dist_mean, str_mut_dist_std, str_cross_sampler_b;
	string s = pt.get<std::string>("abcde.mean", "");
	boost::split(str_mean, s, boost::is_any_of(";"));
	for (int i = 0; i < str_mean.size(); i++)
	{
		mean.push_back(stod(str_mean[i]));
		sample_mean.push_back(stod(str_mean[i]));
	}
	s = pt.get<std::string>("abcde.std", "");
	boost::split(str_std, s, boost::is_any_of(";"));
	for (int i = 0; i < str_std.size() && s != ""; i++)
	{
		std.push_back(stod(str_std[i]));
		sample_std.push_back(stod(str_std[i]));
	}
	s = pt.get<std::string>("abcde.hbound", "");
	boost::split(str_hbound, s, boost::is_any_of(";"));
	for (int i = 0; i < str_hbound.size() && s != ""; i++)
	{
		hbound.push_back(stod(str_hbound[i]));

	}
	s = pt.get<std::string>("abcde.lbound", "");
	boost::split(str_lbound, s, boost::is_any_of(";"));
	for (int i = 0; i < str_lbound.size() && s != ""; i++)
	{
		lbound.push_back(stod(str_lbound[i]));
	}
	s = pt.get<std::string>("abcde.dtype", "");
	boost::split(str_dtype, s, boost::is_any_of(";"));
	for (int i = 0; i < str_dtype.size() && s != ""; i++)
	{
		dtype.push_back(stoi(str_dtype[i]));
	}
	s = pt.get<std::string>("abcde.mut_dist_mean", "");
	boost::split(str_mut_dist_mean, s, boost::is_any_of(";"));
	for (int i = 0; i < str_mut_dist_mean.size() && s != ""; i++)
	{
			mut_dist_mean.push_back(stod(str_mut_dist_mean[i]));
	}
	s = pt.get<std::string>("abcde.mut_dist_std", "");
	boost::split(str_mut_dist_std, s, boost::is_any_of(";"));
	for (int i = 0; i < str_mut_dist_std.size() && s != ""; i++)
	{
		mut_dist_std.push_back(stod(str_mut_dist_std[i]));
	}
	s = pt.get<std::string>("abcde.cross_dist_mean", "");
	boost::split(str_cross_dist_mean, s, boost::is_any_of(";"));
	for (int i = 0; i < str_cross_dist_mean.size() && s != ""; i++)
	{
		cross_dist_mean.push_back(stod(str_cross_dist_mean[i]));
	}
	s = pt.get<std::string>("abcde.cross_dist_std", "");
	boost::split(str_cross_dist_std, s, boost::is_any_of(";"));
	for (int i = 0; i < str_cross_dist_std.size() && s != ""; i++)
	{
		cross_dist_std.push_back(stod(str_cross_dist_std[i]));
	}
	
	s = pt.get<std::string>("abcde.cross_sampler_b", "");
	boost::split(str_cross_sampler_b, s, boost::is_any_of(";"));
	for (int i = 0; i < str_cross_sampler_b.size() && s != ""; i++)
	{
		cross_sampler_b.push_back(stod(str_cross_sampler_b[i]));
	}
	
}

Distribution::Thetha Abcde::mutation(int index)
{
	ofstream logfile("log_mutation.txt", std::ios::app);
	if (print_add_log) logfile << "index = " << index << endl;
	Distribution::Thetha _curr_thetha = posterior.thetha[index];
	for (int i = 0; i < count_opt_param; i++)
	{
		if (print_add_log)
		{
			logfile << "param[" << i << "] = " << i << endl;
			logfile << "prev param = " << _curr_thetha.param[i] << endl;
		}
		_curr_thetha.param[i] = _curr_thetha.param[i] + generator.prior_distribution(Distribution::TYPE_DISTR::NORM, mut_dist_mean[0], mut_dist_std[0]);// another mean for 
		if (print_add_log) logfile << "new param = " << _curr_thetha.param[i] << endl;
	}
	if (print_add_log) logfile << "prev delta = " << _curr_thetha.delta << endl;
	_curr_thetha.delta = _curr_thetha.delta + generator.prior_distribution(Distribution::TYPE_DISTR::EXPON, mut_dist_mean[1]);
	if (print_add_log) logfile << "new delta = " << _curr_thetha.delta << endl;
	logfile.close();
	return _curr_thetha;
}

double Abcde::set_bounds(double x, double _lbound, double _hbound)
{
	double alpha, beta, q;
	alpha = (_hbound + _lbound) / 2.0;
	beta = (_hbound - _lbound) / 2.0;
	if (bounds_crossing_mode == BOUNDS_CROSSING_MODE::SIN)
		q = alpha + beta * sin(x);
	else
		q = alpha + beta * tanh(x);
	return q;
}

Distribution::Thetha Abcde::bounds(Distribution::Thetha _curr_thetha)
{
	Distribution::Thetha thetha;
	for (int i = 0; i < count_opt_param; i++)
	{
		thetha.param.push_back(set_bounds(_curr_thetha.param[i], lbound[i], hbound[i]));
	}
	return thetha;
}
Distribution::Thetha Abcde::crossover(int index)
{
	ofstream logfile("log_crossover.txt", std::ios::app);
	double si_1 = generator.prior_distribution(Distribution::TYPE_DISTR::NORM_WITH_PARAM, cross_dist_mean[0], cross_dist_std[0]), si_2 = generator.prior_distribution(Distribution::TYPE_DISTR::NORM_WITH_PARAM, cross_dist_mean[1], cross_dist_std[1]), b = generator.prior_distribution(Distribution::TYPE_DISTR::NORM_WITH_PARAM, cross_dist_mean[2], cross_dist_std[2]);
	double tmp = max(abs(si_1), abs(si_2));
	si_2 = min(abs(si_1), abs(si_2));
	si_1 = tmp;
	Distribution::Thetha thetha_b, thetha_m, thetha_n, _curr_thetha;
	int m_index, n_index;
	m_index = rand() % (count_iter - 1);
	n_index = rand() % (count_iter - 1);
	if (print_add_log)
	{
		logfile << "index = " << index << endl;
		logfile << "si_1 = " << si_1 << endl;
		logfile << "si_2 = " << si_2 << endl;
		logfile << "b = " << b << endl;
	}

	while (m_index == index)
	{
		m_index = rand() % (count_iter - 1);
	}
	while (m_index == index || m_index == n_index)
	{
		n_index = rand() % (count_iter - 1);
	}

	if (print_add_log)
	{
		logfile << "m_index = " << m_index << endl;
		logfile << "n_index = " << n_index << endl;
	}
	vector<int> a(count_iter); //= {0,1,..., count_iter - 1}
	vector<int> c(count_iter); // = {}
	for (int j = 0; j < count_iter; j++) {
		a[j] = j;
	}
	int len_group_c = rand() % count_iter;
	for (int i = 0; i < len_group_c;i++)
	{
		c[i] = a[rand() % a.size()];
	}


	
	thetha_b = get_prev_iter_with_weight();
	//ïîëó÷èòü â òåòå ñðåäíåå àðèôìåòè÷åñêîå òåò èç ãðóïï à

	thetha_m = posterior.thetha[a[0]];
	for (int i = 0;i < count_iter - len_group_c; i++)
	{

		thetha_m += posterior.thetha[a[i]];
	}
	thetha_m /= count_iter - len_group_c;

	//ïîëó÷èòü â òåòå ñðåäíåå àðèôìåòè÷åñêîå òåò èç ãðóïï b
	thetha_n = 0;
	for (int i = 0;i < len_group_c; i++)
	{
		thetha_n += posterior.thetha[b[i]];
	}
	thetha_n /= len_group_c;

	_curr_thetha = posterior.thetha[index];
	for (int i = 0; i < count_opt_param; i++)
	{
		if (print_add_log)
		{
			logfile << "param[" << i << "] = " << i << endl;
			logfile << "prev param = " << _curr_thetha.param[i] << endl;
		}
		_curr_thetha.param[i] = ((double)_curr_thetha.param[i] + si_1 * ((double)thetha_m.param[i] - (double)thetha_n.param[i])) ;// +si_2 * ((double)thetha_b.param[i] - (double)_curr_thetha.param[i]) + (double)cross_sampler_b[i] * generator.prior_distribution(Distribution::NORM_WITH_PARAM, mean[i], std[i]));
		if (print_add_log) 	logfile << "new param = " << _curr_thetha.param[i] << endl;

	}
	if (print_add_log) logfile << "prev delta = " << _curr_thetha.delta << endl;
	_curr_thetha.delta = _curr_thetha.delta + si_1 * (thetha_m.delta - thetha_n.delta) + b;// +si_2 * (thetha_b.delta - _curr_thetha.delta) + b;
	if (print_add_log) logfile << "new delta = " << _curr_thetha.delta << endl;
	logfile.close();
	return _curr_thetha;
}
double Abcde::get_statistics(Parametrs::MODE _mode,  double _error, int i)
{
	ofstream logfile("log_alpha.txt", std::ios::app);
	double psi_curr, psi_prev, _std;
	if (print_add_log) logfile << "index = " << i << endl;
	if (_mode == Parametrs::MODE::INIT)
	{
		if (print_add_log)
		{
			logfile << "INIT" << endl;
			logfile << "curr_error(x) = " << _error << endl;
			logfile << "curr_delta(std) = " << set_bounds(curr_thetha.delta, 0.005, 5.0) << endl;
		}
		psi_curr = generator.kernel_function(Distribution::TYPE_DISTR::NORM_WITH_PARAM, _error, 0.0, set_bounds(curr_thetha.delta, 0.005, 5.0));
		if (print_add_log) 
		{
			logfile << "curr_psi = " << psi_curr << endl;
			logfile << "prev_error(x) = " << posterior.error[i] << endl;
			logfile << "prev_delta(std) = " << set_bounds(posterior.thetha[i].delta, 0.005, 5.0) << endl;
		}
	    psi_prev = generator.kernel_function(Distribution::TYPE_DISTR::NORM_WITH_PARAM, posterior.error[i], 0.0, set_bounds(posterior.thetha[i].delta, 0.005, 5.0));
		if (print_add_log) logfile << "prev_psi = " << psi_prev << endl;
	}
	else
	{
		if (print_add_log)
		{
			logfile << "ONE DELTA" << endl;
			logfile << "curr_error(x) = " << _error << endl;
			logfile << "curr_delta(std) = " << set_bounds(posterior.delta_one, 0.005, 5.0) << endl;
		}
        psi_curr = generator.kernel_function(Distribution::TYPE_DISTR::NORM_WITH_PARAM, _error, 0.0, set_bounds(posterior.delta_one, 0.005, 5.0));
		if (print_add_log)
		{
			logfile << "curr_psi = " << psi_curr << endl;
			logfile << "prev_error(x) = " << posterior.error[i] << endl;
			logfile << "prev_delta(std) = " << set_bounds(posterior.delta_one, 0.005, 5.0) << endl;
		}
	    psi_prev = generator.kernel_function(Distribution::TYPE_DISTR::NORM_WITH_PARAM, posterior.error[i], 0.0, set_bounds(posterior.delta_one, 0.005, 5.0));
		if (print_add_log) logfile << "prev_psi = " << psi_prev << endl;
	}
	double curr_kernel_func = 1.0, prev_kernel_func = 1.0;
	for (int j = 0; j < count_opt_param; j++)
	{
		if (print_add_log)
		{
			logfile << "param[" << j << "] = " << endl;
			logfile << "curr_param(x) = " << set_bounds(curr_thetha.param[j], lbound[j], hbound[j]) << endl;
			logfile << "prev_param(mean) = " << set_bounds(posterior.thetha[i].param[j], lbound[j], hbound[j]) << endl;
			logfile << "std = " << sample_std[j] << endl;
		}
		_std = sample_std[j];
		_std = min(max(_std, 1.0 / count_opt_param), abs(hbound[j] - lbound[j]) / count_opt_param);
		if (print_add_log) logfile << "final std = " << _std << endl;

		curr_kernel_func *= generator.kernel_function(Distribution::TYPE_DISTR::NORM_WITH_PARAM, set_bounds(curr_thetha.param[j], lbound[j], hbound[j]), set_bounds(posterior.thetha[i].param[j], lbound[j], hbound[j]), _std);
		if (print_add_log)
		{
			logfile << "curr_kernel_func = " << curr_kernel_func << endl;
			logfile << "prev_param(x) = " << set_bounds(posterior.thetha[i].param[j], lbound[j], hbound[j]) << endl;
			logfile << "curr_param(mean) = " << set_bounds(curr_thetha.param[j], lbound[j], hbound[j]) << endl;
			logfile << "std = " << sample_std[j] << endl;
			logfile << "final std = " << _std << endl;
		}
		prev_kernel_func *= generator.kernel_function(Distribution::TYPE_DISTR::NORM_WITH_PARAM, set_bounds(posterior.thetha[i].param[j], lbound[j], hbound[j]), set_bounds(curr_thetha.param[j], lbound[j], hbound[j]), _std);
		if (print_add_log) logfile << "prev_kernel_func = " << prev_kernel_func << endl;
	}
	double curr_alpha = curr_kernel_func * psi_curr;
	if (print_add_log) logfile << "curr_alpha = " << curr_alpha << endl;
	double prev_alpha = prev_kernel_func * psi_prev;
	if (print_add_log) logfile << "prev_alpha = " << prev_alpha << endl;
	if (prev_alpha == 0.0)
	{
		if (print_add_log) logfile << "alpha = " << 0.5 << endl;
		return 0.5;
	}
	double alpha = curr_alpha / prev_alpha;
	if (print_add_log) logfile << "alpha = " << alpha << endl;
	logfile.close();
	return alpha;
}

bool Abcde::accept_alpha(double alpha)
{
	std::random_device rd;
	std::uniform_real_distribution<> dist(0.0, 1.0);
	std::mt19937 engine(rd());
	double value = dist(engine);
	if (value > 1.0 - alpha)
		return true;
	return false;
}

void Abcde::init_posterior()
{
	posterior.thetha = new Distribution::Thetha[count_iter];
	posterior.w = new double[count_iter];
	posterior.error = new double[count_iter];
	new_posterior.thetha = new Distribution::Thetha[count_iter];
	new_posterior.w = new double[count_iter];
	new_posterior.error = new double[count_iter];
}
