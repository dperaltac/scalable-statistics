#include <cstring>

#include "TimeLog.h"

using namespace std;


TimeLog::TimeLog()
{
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	time_current = MPI_Wtime();
	ismaster = (rank == 0);
}


TimeLog::TimeLog(double time_init)
{
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	time_current = time_init;
	ismaster = (rank == 0);
}


TimeLog::TimeLog(const TimeLog &o)
{
	time_current = o.time_current;
	ismaster = o.ismaster;
	times = o.times;
	messages = o.messages;
	memory_phys = o.memory_phys;
	memory_swap = o.memory_swap;
}


TimeLog & TimeLog::operator= (const TimeLog &o)
{
	if(&o != this)
	{
		time_current = o.time_current;
		ismaster = o.ismaster;
		times = o.times;
		messages = o.messages;
		memory_phys = o.memory_phys;
		memory_swap = o.memory_swap;
	}
	
	return *this;
}


void TimeLog::addTimeStamp(const std::string &message)
{
	addManualTimeStamp(message, MPI_Wtime() - time_current);
}


void TimeLog::addTimeStampMinus(const std::string &message, double time_minus)
{
	addManualTimeStamp(message, MPI_Wtime() - time_current - time_minus);
}


void TimeLog::addManualTimeStamp(const std::string &message, double time_manual)
{
	if(ismaster)
	{
		unsigned long mem_phys = 0, mem_swap = 0;
		
		times.push_back(time_manual);
		getTotalMemoryUsed(mem_swap, mem_phys);
		memory_swap.push_back(mem_swap);
		memory_phys.push_back(mem_phys);
		
		messages.push_back(message);
		time_current = MPI_Wtime();
	}
}


void TimeLog::reset()
{
	time_current = MPI_Wtime();
}

double TimeLog::getLastValue() const
{
	return times[times.size()-1];
}

void TimeLog::addSyncTimeStamp(const std::string &message)
{
	MPI_Barrier(MPI_COMM_WORLD);
	addTimeStamp(message);
}

void TimeLog::addSyncManualTimeStamp(const std::string &message, double time_manual)
{
	MPI_Barrier(MPI_COMM_WORLD);
	addManualTimeStamp(message, time_manual);
}


std::ostream& operator<<(std::ostream& out, const TimeLog &tl)
{
	out << "Message,Time,PhysicalMem_KB,VirtualMem_KB" << endl;
	
	for(unsigned int i=0; i < tl.messages.size(); i++)
		out << tl.messages[i] << "," << tl.times[i] << "," << tl.memory_phys[i] << "," << tl.memory_swap[i] << endl;
	
	return out;
}


void TimeLog::saveCSV(const string &filename)
{
	std::ofstream mfile(filename.c_str());

	if(!mfile)
	{
		std::cerr << "TimeLog::saveCSV: Error when opening file " << filename  << " for output " << std::endl;
		return;
	}

	mfile << *this << endl;
	mfile.close();
}


long TimeLog::parseLine(char* line){
	int i = strlen(line);
	const char* p = line;
	while (*p <'0' || *p > '9') p++;
	line[i-3] = '\0';
	return atol(p);
}

void TimeLog::getTotalMemoryUsed(unsigned long &mem_virtual, unsigned long &mem_physical)
{
	FILE* file = fopen("/proc/self/status", "r");
	char line[128];
	
	while (fgets(line, 128, file) != NULL){
		if (strncmp(line, "VmSize:", 7) == 0)
			mem_virtual = parseLine(line);
		else if (strncmp(line, "VmRSS:", 6) == 0)
			mem_physical = parseLine(line);
	}
	fclose(file);
}


