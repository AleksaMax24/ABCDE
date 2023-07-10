#pragma once
#include "pch.h"

class Run_manager
{
public:
	Run_manager() {}
	enum STATE
	{
		INIT = 0,
		RUN_APPROXIMATE,
		RUN,
		END
	};
	int iter;
	int index_thetha;
	int state;
	string log_file = "log_manager_file.ini";
	pt::ptree propTree;
	void create_log_file(int _state, Distribution::Posterior& posterior, Distribution::Posterior& new_posterior, double norm_error, int iter, int index_thetha, int count_opt_param);

	void read_log_file(Distribution::Posterior& posterior, Distribution::Posterior& new_posterior, double& norm_error, int count_iter, int count_opt_param);

	void change_state(int _state);
	
	void change_delta(double delta);
};
