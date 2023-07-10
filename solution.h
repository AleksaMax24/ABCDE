#pragma once
#include "pch.h"

class Solution
{
private:
	Abcde main_model;
	Deep aux_model;
	double alpha;
	Parametrs param;
	Run_manager manager;
    int neural_network_trained_flag = 0;
public:
	Solution(const Abcde& _main_model, const Deep& _aux_model, const Parametrs& _param);

	void run_init(int iter, int index_thetha);

	void run_approximate(int iter, int index_thetha);

	void run(int iter, int index_thetha);

	void print_log(int iter);

	void copy_posterior( Distribution::Posterior& posterior_to,  Distribution::Posterior& posterior_from);

	void run_manager();
    void train_neural_network();
};
