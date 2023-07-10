#include "pch.h"
#include "ABCDE_SSM/summary_statistics/summary_statistics_computer.h"
#include "ABCDE_SSM/summary_statistics/NeuralNetwork.h"

Solution::Solution(const Abcde& _main_model, const Deep& _aux_model, const Parametrs& _param) {
	main_model = _main_model;
	aux_model = _aux_model;
	param = _param;
	alpha = 0.0;
}

inline void Solution::copy_posterior(Distribution::Posterior& posterior_to, Distribution::Posterior& posterior_from)
{
	for (int i = 0; i < main_model.count_iter; i++)
	{
		posterior_to.thetha[i] = posterior_from.thetha[i];
		posterior_to.w[i] = posterior_from.w[i];
		posterior_to.error[i] = posterior_from.error[i];
	}
}

void Solution::train_neural_network() {
    if (aux_model.errorMode != ErrorMode::CNN && aux_model.errorMode != ErrorMode::DNN && neural_network_trained_flag == 0)
        return;

    int train_neural_number = aux_model.train_neural_number;
    std::vector<std::vector<double>> trainData;
    std::vector<std::vector<double>> trainLabels;

    for (int i = 0; i < train_neural_number; ++i) {
        cout << "Train iter is: " << i << endl;
        Distribution::Thetha thetha = main_model.generate_vector_param(Distribution::NORM_WITH_PARAM);

        aux_model.create_tmp_deep_ini_file();
        int seed = main_model.generator.generate_seed();
        aux_model.prepare_tmp_deep_ini_file(main_model.bounds(thetha), main_model.dtype, seed);

        double error = aux_model.run(-1, i, seed);
        SSM_DATA* ssmData = aux_model.last_ssm_data;

        if (ssmData == nullptr || ssmData->ssmData.size() == 0) {
            i--;
            continue;
        }

        trainData.push_back(ssmData->ssmData);
        trainLabels.push_back(thetha.param);
    }

    neural_network_trained_flag = 1;

    switch (aux_model.errorMode) {
        case ErrorMode::CNN:
            cout << "start train CNN" << trainData[0].size() << endl;
            aux_model.neuralNetworkDistribution = new CNN(trainData[0].size(), trainLabels[0].size());
            break;
        case ErrorMode::DNN:
            cout << "start train DNN" << endl;
            aux_model.neuralNetworkDistribution = new DNN(trainData[0].size(), trainLabels[0].size());
            break;
    }

    aux_model.neuralNetworkDistribution->train(trainData, trainLabels);
}

void Solution::run_manager()
{
	manager.read_log_file(main_model.posterior, main_model.new_posterior, main_model.norm_error, main_model.count_iter, main_model.count_opt_param);
    train_neural_network();
	switch (manager.state)
	{
	case Run_manager::STATE::INIT:
		run_init(manager.iter + 1, manager.index_thetha);
		break;
	case Run_manager::STATE::RUN_APPROXIMATE:
		run_approximate(manager.iter + 1, manager.index_thetha);
		break;
	case Run_manager::STATE::RUN:
		run(manager.iter + 1, manager.index_thetha);
		break;
	}
}

void Solution::run_init(int iter, int index_thetha)
{
    cout << "start run_init" << endl;
#ifdef MPIZE
	int tag = 0;
	int rank;
#endif
	int size = 1;
#ifdef MPIZE
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	cout << "This rank is " << rank << endl;
	cout << "This size is " << size << endl;
	MPI_Status status;
	if (rank == 0)
	{
#endif
		ofstream out("log_iteration.txt", std::ios::app);
		out << "INIT" << endl;
		vector<Distribution::Thetha> all_thetha;
		for (int j = 0; j < size; j++)
		{
			vector<vector<double>> _param;
			for (int i = 0; i < main_model.count_iter / size; i++)
			{
				main_model.curr_thetha = main_model.generate_vector_param(Distribution::NORM_WITH_PARAM);
				_param.push_back(main_model.curr_thetha.param);
				all_thetha.push_back(main_model.curr_thetha);
			}
#ifdef MPIZE
			if (j != 0)
			{
				for (int k = 0; k < main_model.count_iter / size; k++)
					MPI_Send(&_param[k].front(), main_model.count_opt_param, MPI_DOUBLE, j, tag, MPI_COMM_WORLD);
			}
#endif
		}
		for (int i = 0; i < main_model.count_iter / size; i++)
		{
			double error;
			main_model.curr_thetha = all_thetha[i];
			out << "iteration = " << -1 << endl;
			out << "element number = " << i << endl;
			for (int s = 0; s < main_model.count_opt_param; s++)
				out << main_model.curr_thetha.param[s] << endl;
			aux_model.create_tmp_deep_ini_file();
			int seed = main_model.generator.generate_seed();
			aux_model.prepare_tmp_deep_ini_file(main_model.bounds(main_model.curr_thetha), main_model.dtype, seed);
			error = aux_model.run(-1, i, seed);
			if (i == 0)
				main_model.norm_error = error;
			main_model.posterior.thetha[i] = main_model.curr_thetha;
			main_model.posterior.w[i] = 1.0 / main_model.count_iter;
			main_model.posterior.error[i] = error / main_model.norm_error;
			main_model.new_posterior.thetha[i] = main_model.curr_thetha;
			main_model.new_posterior.w[i] = 1.0 / main_model.count_iter;
			main_model.new_posterior.error[i] = error / main_model.norm_error;
			main_model.posterior.thetha[i].delta = main_model.new_posterior.thetha[i].delta = main_model.generator.prior_distribution(Distribution::TYPE_DISTR::EXPON, 0.005);
			out << "delta = " << main_model.posterior.thetha[i].delta << endl;
			out << "error = " << error / main_model.norm_error << endl;
		}
#ifdef MPIZE
		for (int j = 1; j < size; j++)
		{
			vector<double> error;
			error.resize(main_model.count_iter / size);
			MPI_Recv(&error[0], main_model.count_iter / size, MPI_DOUBLE, j, tag, MPI_COMM_WORLD, &status);
			for (int i = 0; i < main_model.count_iter / size; i++)
			{
				main_model.posterior.thetha[j * main_model.count_iter / (size)+i] = all_thetha[j * main_model.count_iter / (size)+i];
				out << "iteration = " << -1 << endl;
				out << "element number = " << j * main_model.count_iter / (size) + i << endl;
				for (int s = 0; s < main_model.count_opt_param; s++)
					out << main_model.posterior.thetha[j * main_model.count_iter / (size)+i].param[s] << endl;
				main_model.posterior.w[j * main_model.count_iter / (size)+i] = 1.0 / main_model.count_iter;
				main_model.posterior.error[j * main_model.count_iter / (size)+i] = error[i] / main_model.norm_error;
				main_model.new_posterior.thetha[j * main_model.count_iter / (size)+i] = all_thetha[j * main_model.count_iter / (size)+i];
				main_model.new_posterior.w[j * main_model.count_iter / (size)+i] = 1.0 / main_model.count_iter;
				main_model.new_posterior.error[j * main_model.count_iter / (size)+i] = error[i] / main_model.norm_error;
				main_model.posterior.thetha[j * main_model.count_iter / (size)+i].delta = main_model.new_posterior.thetha[j * main_model.count_iter / (size)+i].delta = main_model.generator.prior_distribution(Distribution::TYPE_DISTR::EXPON, 0.005);
				out << "delta = " << main_model.posterior.thetha[j * main_model.count_iter / (size)+i].delta << endl;
				out << "error = " << error[i] / main_model.norm_error << endl;
			}
		}
#endif
		for (int i = 0; i < main_model.count_iter; i++)
			manager.create_log_file(manager.state, main_model.posterior, main_model.new_posterior, main_model.norm_error, -1, i, main_model.count_opt_param);
		main_model.set_sample_dist_param();
		main_model.get_index_best();
		print_log(-1);
		manager.state = Run_manager::STATE::RUN_APPROXIMATE;
		manager.change_state(manager.state);
		out.close();
		run_approximate(0, 0);
#ifdef MPIZE
	}
	else
	{
		vector<vector<double>> _param(main_model.count_iter / size, vector<double>(main_model.count_opt_param));
		vector<double>error;
		for (int k = 0; k < main_model.count_iter / size; k++)
			MPI_Recv(&_param[k][0], main_model.count_opt_param, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
		for (int i = 0; i < main_model.count_iter / size; i++)
		{
			Distribution::Thetha curr_thetha;
			curr_thetha.param = _param[i];
			int seed = main_model.generator.generate_seed();
			aux_model.create_tmp_deep_ini_file();
			aux_model.prepare_tmp_deep_ini_file(main_model.bounds(curr_thetha), main_model.dtype, seed);
			error.push_back(aux_model.run(-1, rank * main_model.count_iter / (size)+i, seed));
		}
		MPI_Send(&error[0], main_model.count_iter / size, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
		run_approximate(0, 0);
	}
#endif
}

void Solution::run_approximate(int iter, int index_thetha)
{
    cout << "start run_approximate" << endl;
#ifdef MPIZE
	int tag = 0;
	int rank;
#endif
	int size = 1;
#ifdef MPIZE
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Status status;
	if (rank == 0)
	{
#endif
		ofstream out("log_iteration.txt", std::ios::app);
		out << "RUN_APPROXIMATE" << endl;
		for (int t = iter; t < main_model.start_iter; t++)
		{
			vector<Distribution::Thetha> all_thetha;
			for (int j = 0; j < size; j++)
			{
				vector<vector<double>> _param;

				for (int i = 0; i < main_model.count_iter / size; i++)
				{
					if (main_model.crossing_mode == Abcde::CROSSING_MODE::ALL)
					{
						double choice = main_model.generator.prior_distribution(Distribution::TYPE_DISTR::RANDOM, 0.0, 1.0);
						if (choice < 0.05)
						{
							main_model.curr_thetha = main_model.mutation(i + j * main_model.count_iter / size);
						}
						else
						{
							main_model.curr_thetha = main_model.crossover(i + j * main_model.count_iter / size);
						}
					}
					else if (main_model.crossing_mode == Abcde::CROSSING_MODE::ONLY_CROSSOVER)
						main_model.curr_thetha = main_model.crossover(i + j * main_model.count_iter / size);
					else if (main_model.crossing_mode == Abcde::CROSSING_MODE::ONLY_MUTATION)
						main_model.curr_thetha = main_model.mutation(i + j * main_model.count_iter / size);


					_param.push_back(main_model.curr_thetha.param);
					all_thetha.push_back(main_model.curr_thetha);
				}
#ifdef MPIZE
				if (j != 0)
				{
					for (int k = 0; k < main_model.count_iter / size; k++)
						MPI_Send(&_param[k].front(), main_model.count_opt_param, MPI_DOUBLE, j, tag, MPI_COMM_WORLD);
				}
#endif
			}
			for (int i = 0; i < main_model.count_iter / (size); i++)
			{
				double error;
				main_model.curr_thetha = all_thetha[i];
				out << "iteration = " << t << endl;
				out << "element number = " << i << endl;
				for (int s = 0; s < main_model.count_opt_param; s++)
					out << main_model.curr_thetha.param[s] << endl;
				out << "delta = " << main_model.curr_thetha.delta << endl;
				int seed = main_model.generator.generate_seed();
				aux_model.create_tmp_deep_ini_file();
				aux_model.prepare_tmp_deep_ini_file(main_model.bounds(main_model.curr_thetha), main_model.dtype, seed);
				error = aux_model.run(t, i, seed);
				out << "error = " << error / main_model.norm_error << endl;
				alpha = main_model.get_statistics(Parametrs::MODE::INIT, error / main_model.norm_error, i);
				out << "original alpha = " << alpha << endl;
				alpha = min(1.0, alpha);
				out << "alpha = " << alpha << endl;
				if (main_model.accept_alpha(alpha))
				{
					out << "accept alpha" << endl;
					main_model.new_posterior.thetha[i] = main_model.curr_thetha;
					main_model.new_posterior.error[i] = error / main_model.norm_error;
				}
				else
					out << "not accept" << endl;
			}
#ifdef MPIZE
			for (int j = 1; j < size; j++)
			{
				vector<double> error(main_model.count_iter / (size));
				MPI_Recv(&error[0], main_model.count_iter / size, MPI_DOUBLE, j, tag, MPI_COMM_WORLD, &status);
				for (int i = 0; i < main_model.count_iter / size; i++)
				{
					main_model.curr_thetha = all_thetha[j * main_model.count_iter / (size)+i];
					out << "iteration = " << t << endl;
					out << "element number = " << j * main_model.count_iter / (size)+i << endl;
					for (int s = 0; s < main_model.count_opt_param; s++)
						out << main_model.curr_thetha.param[s] << endl;
					out << "delta = " << main_model.curr_thetha.delta << endl;
					alpha = main_model.get_statistics(Parametrs::MODE::INIT, error[i] / main_model.norm_error, j * main_model.count_iter / (size)+i);
					out << "error = " << error[i] / main_model.norm_error << endl;
					out << "original alpha = " << alpha << endl;
					alpha = min(1.0, alpha);
					out << "alpha = " << alpha << endl;
					if (main_model.accept_alpha(alpha))
					{
						out << "accept alpha" << endl;
						main_model.new_posterior.thetha[j * main_model.count_iter / (size)+i] = main_model.curr_thetha;
						main_model.new_posterior.error[j * main_model.count_iter / (size)+i] = error[i] / main_model.norm_error;
					}
					else
						out << "not accept" << endl;
				}
			}
#endif	
			main_model.get_index_best();

			main_model.update_posterior();//перерасчет весов
			copy_posterior(main_model.posterior, main_model.new_posterior); // перестановка
			main_model.set_sample_dist_param();

			for (int i = 0; i < main_model.count_iter; i++)
				manager.create_log_file(manager.state, main_model.posterior, main_model.new_posterior, main_model.norm_error, t, i, main_model.count_opt_param);
			print_log(t);
		}
		double s = 0.0;
		for (int i = 0; i < main_model.count_iter; i++)
		{
			s += main_model.posterior.thetha[i].delta;
		}
		main_model.posterior.delta_one = s / main_model.count_iter;
		main_model.new_posterior.delta_one = main_model.posterior.delta_one;
		manager.state = Run_manager::STATE::RUN;
		manager.change_state(manager.state);
		manager.change_delta(main_model.posterior.delta_one);
		out.close();
		run(0, 0);
#ifdef MPIZE
	}
	else
	{
		for (int t = iter; t < main_model.start_iter; t++)
		{
			vector<vector<double>> _param(main_model.count_iter / size, vector<double>(main_model.count_opt_param));
			vector<double> error;
			for (int k = 0; k < main_model.count_iter / size; k++)
				MPI_Recv(&_param[k][0], main_model.count_opt_param, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
			for (int i = 0; i < main_model.count_iter / size; i++)
			{
				Distribution::Thetha curr_thetha;
				curr_thetha.param = _param[i];
				int seed = main_model.generator.generate_seed();
				aux_model.create_tmp_deep_ini_file();
				aux_model.prepare_tmp_deep_ini_file(main_model.bounds(curr_thetha), main_model.dtype, seed);
				error.push_back(aux_model.run(t, rank * main_model.count_iter / (size)+i, seed));
			}
			MPI_Send(&error[0], main_model.count_iter / size, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);

		}

		run(0, 0);
	}
#endif
}

void Solution::run(int iter, int index_thetha)
{
#ifdef MPIZE
	int tag = 0;
	int rank;
#endif
	int size = 1;
#ifdef MPIZE
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;
	if (rank == 0)
	{
#endif
		ofstream out("log_iteration.txt", std::ios::app);
		out << "RUN" << endl;
		for (int t = iter; t < main_model.t; t++)
		{
			vector<Distribution::Thetha> all_thetha;
			for (int j = 0; j < size; j++)
			{
				vector<vector<double>> _param;

				for (int i = 0; i < main_model.count_iter / size; i++)
				{
					if (main_model.crossing_mode == Abcde::CROSSING_MODE::ALL)
					{
						double choice = main_model.generator.prior_distribution(Distribution::TYPE_DISTR::RANDOM, 0.0, 1.0);
						if (choice < 0.05)
						{
							main_model.curr_thetha = main_model.mutation(i + j * main_model.count_iter / size);
						}
						else
						{
							main_model.curr_thetha = main_model.crossover(i + j * main_model.count_iter / size);
						}
					}
					else if (main_model.crossing_mode == Abcde::CROSSING_MODE::ONLY_CROSSOVER)
						main_model.curr_thetha = main_model.crossover(i + j * main_model.count_iter / size);
					else if (main_model.crossing_mode == Abcde::CROSSING_MODE::ONLY_MUTATION)
						main_model.curr_thetha = main_model.mutation(i + j * main_model.count_iter / size);
					
					_param.push_back(main_model.curr_thetha.param);
					all_thetha.push_back(main_model.curr_thetha);
				}
#ifdef MPIZE
				if (j != 0)
				{
					for (int k = 0; k < main_model.count_iter / size; k++)
						MPI_Send(&_param[k].front(), main_model.count_opt_param, MPI_DOUBLE, j, tag, MPI_COMM_WORLD);
				}
#endif
			}
			for (int i = 0; i < main_model.count_iter / size; i++)
			{
				double error;
				main_model.curr_thetha = all_thetha[i];
				int seed = main_model.generator.generate_seed();
				aux_model.create_tmp_deep_ini_file();
				aux_model.prepare_tmp_deep_ini_file(main_model.bounds(main_model.curr_thetha), main_model.dtype, seed);
				error = aux_model.run(t, i, seed);
				out << "iteration = " << t << endl;
				out << "element number = " << i << endl;
				for (int s = 0; s < main_model.count_opt_param; s++)
					out << main_model.curr_thetha.param[s] << endl;
				out << "delta = " << main_model.curr_thetha.delta << endl;

				alpha = main_model.get_statistics(Parametrs::MODE::AUX, error / main_model.norm_error, i);
				out << "error = " << error / main_model.norm_error << endl;

				out << "original alpha = " << alpha << endl;
				alpha = min(1.0, alpha);
				out << "alpha = " << alpha << endl;
				if (main_model.accept_alpha(alpha))
				{
					out << "accept alpha" << endl;
					main_model.new_posterior.thetha[i] = main_model.curr_thetha;
					main_model.new_posterior.error[i] = error / main_model.norm_error;
				}
				else
					out << "not accept" << endl;
			}
#ifdef MPIZE
			for (int j = 1; j < size; j++)
			{
				vector<double> error(main_model.count_iter / (size));
				MPI_Recv(&error[0], main_model.count_iter / size, MPI_DOUBLE, j, tag, MPI_COMM_WORLD, &status);
				for (int i = 0; i < main_model.count_iter / size; i++)
				{
					main_model.curr_thetha = all_thetha[j * main_model.count_iter / (size)+i];
					out << "iteration = " << t << endl;
					out << "element number = " << j * main_model.count_iter / (size)+i << endl;
					for (int s = 0; s < main_model.count_opt_param; s++)
						out << main_model.curr_thetha.param[s] << endl;
					out << "delta = " << main_model.curr_thetha.delta << endl;

					alpha = main_model.get_statistics(Parametrs::MODE::AUX, error[i] / main_model.norm_error, j * main_model.count_iter / (size)+i);
					out << "error = " << error[i] / main_model.norm_error << endl;

					out << "original alpha = " << alpha << endl;
					alpha = min(1.0, alpha);
					out << "alpha = " << alpha << endl;
					if (main_model.accept_alpha(alpha))
					{
						out << "accept alpha" << endl;
						main_model.new_posterior.thetha[j * main_model.count_iter / (size)+i] = main_model.curr_thetha;
						main_model.new_posterior.error[j * main_model.count_iter / (size)+i] = error[i] / main_model.norm_error;
					}
					else
						out << "not accept" << endl;
				}
			}
#endif				
			main_model.get_index_best();

			main_model.update_posterior();//перерасчет весов
			copy_posterior(main_model.posterior, main_model.new_posterior);//перестановка
			main_model.set_sample_dist_param();

			for (int i = 0; i < main_model.count_iter; i++)
				manager.create_log_file(manager.state, main_model.posterior, main_model.new_posterior, main_model.norm_error, t, i, main_model.count_opt_param);
			print_log(t);
		}
		manager.state = Run_manager::STATE::END;
		manager.change_state(manager.state);
		out.close();
#ifdef MPIZE
	}
	else
	{
		for (int t = iter; t < main_model.t; t++)
		{
			vector<vector<double>> _param(main_model.count_iter / size, vector<double>(main_model.count_opt_param));
			vector<double> error;
			for (int k = 0; k < main_model.count_iter / size; k++)
				MPI_Recv(&_param[k][0], main_model.count_opt_param, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
			for (int i = 0; i < main_model.count_iter / (size); i++)
			{
				Distribution::Thetha curr_thetha;
				curr_thetha.param = _param[i];
				int seed = main_model.generator.generate_seed();
				aux_model.create_tmp_deep_ini_file();
				aux_model.prepare_tmp_deep_ini_file(main_model.bounds(curr_thetha), main_model.dtype, seed);
				error.push_back(aux_model.run(t, rank * main_model.count_iter / (size)+i, seed));
			}
			MPI_Send(&error[0], main_model.count_iter / size, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
		}
	}
#endif
}

void Solution::print_log(int iter)
{
	ofstream logfile("log_result.txt", std::ios::app);
	logfile << "iteration = " << iter << endl;
	for (int i = 0; i < main_model.count_iter; i++)
	{
		logfile << "element number = " << i << " ";
		for (int j = 0; j < main_model.count_opt_param; j++)
		{
			logfile << "param[" << j << "] = " << main_model.posterior.thetha[i].param[j] << " ";
		}
		logfile << "w = " << main_model.posterior.w[i] << " ";
		logfile << "error = " << main_model.posterior.error[i] << " ";
		logfile << "delta = " << main_model.posterior.thetha[i].delta << " ";
		logfile << endl;
	}
	logfile.close();
}
