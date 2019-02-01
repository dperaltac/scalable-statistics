/**
 * \file    Functions.h
 * \author  Daniel Peralta <daniel.peralta@irc.vib-ugent.be>
 * \version 1.0
 *
 * \section DESCRIPTION_FUNCTIONS
 *
 * Header file defining some useful functions
 */

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include<string>

/**
 * Computes the square power of the given number
 * \param z Number to be multiplied by itself

 * \return \p z to the square power
 */
template<class T>
inline T square(T z) {return z*z;}

/**
 * Checks whether a string can be interpreted as an integer number.
 * \param s Input string
 * 
 * \return true if the string contains an integer, false otherwise.
 */
bool isInteger(const std::string &s);

/**
 * Shows a message in stderr and stops the execution
 * \param message Message that is shown in stderr
 * \param status Return value of the program
 */
void exit(const std::string & message, int status);

/**
 * Prints the time elapsed from \p time_init in seconds
 * \param time_init Time in seconds as returned by MPI_Wtime()
 */
void printTime(double time_init);

/**
 * Prints message \p message, the time elapsed from \p time_tmp in seconds, and the time elapsed from \p time_init in seconds. Also returns the current time.
 * \param message Message to print
 * \param time_tmp Time in seconds as returned by MPI_Wtime(), corresponding to the start of the current chunk that is being measured
 * \param time_init Time in seconds as returned by MPI_Wtime(), corresponding to the start of the execution
 * \return Current time
 */
double printTimeInt(const std::string & message, double time_tmp, double time_init, const std::string &separator = "\t");

#endif
