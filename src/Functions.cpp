#include <iostream>

#include "Functions.h"
#include "mpi.h"

bool isInteger(const std::string &s)
{
	if(s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+')))
		return false;
	
	char * p ;
	strtol(s.c_str(), &p, 10);
	
	return (*p == 0);
}


void exit(const std::string & message, int status)
{
	int flag = 0;
	
	std::cerr << message << std::endl;
	
	MPI_Initialized(&flag);
	
	if(flag)
		MPI_Finalize();
	
	std::exit(status);
}


void printTime(double time_init)
{
	std::cout << "Elapsed time: " << MPI_Wtime() - time_init << std::endl;
}


double printTimeInt(const std::string & message, double time_tmp, double time_init, const std::string &separator)
{
	double time_cur = MPI_Wtime();
	std::cout << message << time_cur - time_tmp << separator << time_cur - time_init << std::endl;
	
	return time_cur;
}

