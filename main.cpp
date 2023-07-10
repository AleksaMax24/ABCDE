#include "pch.h"


int main(int argc, char* argv[])
{
	int numTask, rank;
#ifdef MPIZE
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numTask);
#endif
	Parametrs param;
	param.process_program_options(argc, argv);
	Abcde abcde(param.config_file);
	Deep deep(param.config_file);
	Solution solution(abcde, deep, param);
	solution.run_manager();
#ifdef MPIZE
	MPI_Finalize();
#endif
	return 0;
}
