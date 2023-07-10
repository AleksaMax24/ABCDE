#pragma once
#include "pch.h"

class Parametrs
{
public:
	enum MODE
	{
		INIT,
		AUX
	};
	string config_file;

	void train(const po::variables_map& vm)
	{
		if (vm.count("help"))
		{
			cout << "--help - show help " << endl;
			cout << "--input [file_name] - config_file" << endl;
			cout << "--test [0 - med, 1 - mean] - mode for get delta after training" << endl;

		}
		if (vm.count("input")) {
			config_file = vm["input"].as<std::string>();
		}
		
	}

	void process_program_options(const int ac, const char* const av[])
	{
		po::options_description desc("General options");
		desc.add_options()
			("help,h", "Show help")
			;
		po::options_description solution_desc("Solution options");
		solution_desc.add_options()
			("input,I", po::value<std::string>(), "Input config file")
			
			;
		po::variables_map vm;
		po::parsed_options parsed = po::command_line_parser(ac, av).options(desc).allow_unregistered().run();
		po::store(parsed, vm);
		po::notify(vm);
		desc.add(solution_desc);
		po::store(po::parse_command_line(ac, av, desc), vm);
		train(vm);
	}
};
