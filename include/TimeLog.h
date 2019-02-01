/**
 * \file    TimeLog.h
 * \author  Daniel Peralta <daniel.peralta@irc.vib-ugent.be>
 * \version 1.0
 *
 * \section DESCRIPTION_TIMELOG
 *
 * Defines class TimeLog
 */
#ifndef TIMELOG_H
#define TIMELOG_H

#include <vector>
#include <iostream>
#include <fstream>
#include "mpi.h"

/**
 * \class TimeLog
 * 
 * A simple log that saves a string along with a timestamp. The MPI walltime is used.
 * Only the master process does the logging, and most methods do nothing in the slave processes. However, all processes should call the same methods to ensure a correct synchronization and time measurement.
 */
class TimeLog
{
public:
	
	/** Default Constructor. The current time is taken as initial time. */
	TimeLog();
	
	/**
	 * Constructor
	 * \param time_init The initial time w.r.t. which the timestamps will be measured
	 */
	TimeLog(double time_init);
	
	/**
	 * Copy constructor
	 * \param o Object to copy from
	 */
	TimeLog(const TimeLog &o);
	
	/** Default destructor */
	~TimeLog() {};
	
	/**
	 * Assignment operator
	 * \param o Object to assign from
	 * \return A reference to this object
	 */
	TimeLog & operator= (const TimeLog &o);
	
	/**
	 * Returns the last timestamp value
	 * \return Last timestamp
	 */
	double getLastValue() const;
	
	/**
	 * Adds a new timestamp, labeled with a message
	 * \param message The message assigned to this timestamp
	 * */
	void addTimeStamp(const std::string &message);
	
	/**
	 * Adds a new timestamp, labeled with a message, and substracts \p time_minus to the current time. This can be used to avoid counting some times that are spent in the computation but should not be measured, like I/O.
	 * \param message The message assigned to this timestamp
	 * \param time_minus Time to be substracted
	 * */
	void addTimeStampMinus(const std::string &message, double time_minus);
	
	/**
	 * Adds a new timestamp with the time specified in \p time_manual
	 * \param message The message assigned to this timestamp
	 * \param time_manual Time for the timestamp
	 * */
	void addManualTimeStamp(const std::string &message, double time_manual);
	
	/**
	 * First executes a MPI_Barrier, then adds a new timestamp, labeled with a message
	 * \param message The message assigned to this timestamp
	 * */
	void addSyncTimeStamp(const std::string &message);
	
	/**
	 * Sets the current time as initial time
	 * */
	void reset();
	
	/**
	 * First executes a MPI_Barrier, then adds a new timestamp with the time specified in \p time_manual
	 * \param message The message assigned to this timestamp
	 * \param time_manual Time for the timestamp
	 * */
	void addSyncManualTimeStamp(const std::string &message, double time_manual);
	
	
	/**
	 * Output operator <<
	 * \param out Output stream
	 * \param tl Timelog to print
	 * \return The output stream
	 */
	friend std::ostream& operator<< (std::ostream& out, const TimeLog& tl);
	
	/**
	 * Saves the log to a CSV file
	 * \param filename Name of the output CSV file.
	 */
	void saveCSV(const std::string &filename);
	
private:
	bool ismaster;	///< Whether the current process is master or not
	std::vector<std::string> messages;	///< Contain the messages associated with the timestamps
	std::vector<double> times;	///< Contains the timestamps
	std::vector<unsigned long> memory_swap;	///< Contains the swap consumption
	std::vector<unsigned long> memory_phys;	///< Contains the physical memory consumption
	double time_current;	///< Current walltime 
	
	
	/**
	 * Auxiliary function to parse the lines of /proc/self/status
	 * \param line Line that will be parsed (must end in "kB")
	 * \return Number that is found in the line
	 */
	long parseLine(char* line);
	
	/**
	 * Gets the amount of memory used by the current process
	 * \param mem_swap Output parameter. Amount of swap memory used by the current process (in KB)
	 * \param mem_phys Output parameter. Amount of physical memory used by the current process (in KB)
	 */
	void getTotalMemoryUsed(unsigned long &mem_virtual, unsigned long &mem_physical);
};

#endif
