#include "pch.h"


void Run_manager::change_state(int _state)
{
	propTree.put("param.state", _state);
	propTree.put("param.iter", "0");
	propTree.put("param.index_thetha", "0");
	write_ini(log_file, propTree);
}


void Run_manager::change_delta(double delta)
{
	propTree.put("delta_one", delta);
	write_ini(log_file, propTree);
}

void Run_manager::create_log_file(int _state, Distribution::Posterior& posterior, Distribution::Posterior& new_posterior, double norm_error, int _iter, int _index_thetha, int count_opt_param)
{
	propTree.put("param.state", _state);
	propTree.put("param.iter", _iter);
	propTree.put("param.index_thetha", _index_thetha);
	propTree.put("norm_error", norm_error);
	string keys;
	for (int i = 0; i < count_opt_param; i++)
	{
		keys = "thetha." + to_string(_index_thetha) + "_" + to_string(i);
		propTree.put(keys, posterior.thetha[_index_thetha].param[i]);

		keys = "new_thetha." + to_string(_index_thetha ) + "_" + to_string(i);
		propTree.put(keys, new_posterior.thetha[_index_thetha].param[i]);

		
	}

	keys = "w." + to_string(_index_thetha);
	propTree.put(keys, posterior.w[_index_thetha]);

	keys = "new_w." + to_string(_index_thetha);
	propTree.put(keys, posterior.w[_index_thetha]);


	keys = "error." + to_string(_index_thetha);
	propTree.put(keys, posterior.error[_index_thetha]);

	keys = "new_error." + to_string(_index_thetha);
	propTree.put(keys, new_posterior.error[_index_thetha]);


	keys = "delta." + to_string(_index_thetha);
	propTree.put(keys, posterior.thetha[_index_thetha].delta);

	keys = "new_delta." + to_string(_index_thetha);
	propTree.put(keys, new_posterior.thetha[_index_thetha].delta);
	if (_state == STATE::RUN)
	{
		keys = "delta_one";
		propTree.put(keys, new_posterior.delta_one);
		propTree.put(keys, posterior.delta_one);

	}
	write_ini(log_file, propTree);
}
//count_iter <---> count_thetha
void Run_manager::read_log_file(Distribution::Posterior& posterior, Distribution::Posterior& new_posterior, double &norm_error, int count_iter, int count_opt_param)
{
	ifstream file;
	file.open(log_file);
	if (!file)
	{
		state = INIT;
		iter = 0;
		index_thetha = 0;
		return;
	}
	boost::property_tree::ini_parser::read_ini(log_file, propTree);
	state = stoi(propTree.get<std::string>("param.state"));
	iter = stoi(propTree.get<std::string>("param.iter"));
	index_thetha = stoi(propTree.get<std::string>("param.index_thetha"));
	//norm_error = stod(propTree.get<std::string>("norm_error"));
	string keys;
	for (int i = 0; i < count_iter; i++)
	{
		for(int j = 0; j < count_opt_param; j++)
		{ 
			keys = "thetha." + to_string(i) + "_" + to_string(j);
			posterior.thetha[i].param.push_back(stod(propTree.get<std::string>(keys)));
			keys = "new_thetha." + to_string(i) + "_" + to_string(j);
			new_posterior.thetha[i].param.push_back(stod(propTree.get<std::string>(keys)));
		}

		keys = "w." + to_string(i);
		posterior.w[i] = stod(propTree.get<std::string>(keys));

		keys = "new_w." + to_string(i);
		new_posterior.w[i] = stod(propTree.get<std::string>(keys));

		keys = "error." + to_string(i);
		posterior.error[i] = stod(propTree.get<std::string>(keys));

		keys = "new_error." + to_string(i);
		new_posterior.error[i] = stod(propTree.get<std::string>(keys));

		keys = "delta." + to_string(i);
		posterior.thetha[i].delta = stod(propTree.get<std::string>(keys));

		keys = "new_delta." + to_string(i);
		new_posterior.thetha[i].delta = stod(propTree.get<std::string>(keys));

		if (state == STATE::RUN)
		{
			keys = "delta_one";
			posterior.delta_one = stod(propTree.get<std::string>(keys));
			new_posterior.delta_one = stod(propTree.get<std::string>(keys));
		}
	}
}
