/**
 * \file    BuildFeatureSimilarityMatrixByClass.cpp
 * \author  Daniel Peralta <daniel.peralta@irc.vib-ugent.be>
 * \version 1.0
 *
 * Main program for extracting feature similarity matrices, mainly based on covariance and other measures.
 * The computation is carried out in parallel, using numerically stable algorithms.
 * The statistics are computed per-class, including the following information for each class:
 * - Average value
 * - A priori probability (proportion of instances)
 * - Covariance matrix (including variance)
 * 
 * The class field should be the last column of the file, encoded either as integers starting from 0, either as strings if --classfile is provided.
 * The optional parameter --classfile allows to pass a file containing the names of each class. This allows to group several integers into the same class. If provided, instances with classes that are not in the file will not be considered.
 * 
 */

#include <iostream>
#include <cmath>
#include <unistd.h>
#include <vector>
#include <map>
#include <stack>
#include "mpi.h"
#include "omp.h"

#include "Matrix.h"
#include "Tensor.h"
#include "Hyperplane.h"
#include "Hyperplane2D.h"
#include "Functions.h"
#include "TimeLog.h"
using namespace std;

int MIN_AGGREGATION_NUM = 128;	///< Base case of recursive computation; it is the maximum size of a chunk for it to be processed in a sequential way.
const int MASTER_PROCESS = 0;	///< MPI rank of the master process

// Global variables to store the classes of the dataset
/// \todo can I replace NUM_CLASSES by class_names.size() or by the dimension of the arrays?
int NUM_CLASSES = -1;	///< Number of classes

// More flags
bool DEBUG = false;	///< DEBUG flag
bool MEASURE_TIMES = false;	///< Whether to measure and output times or not

/**
 * Aggregates two T (total) values
 * \param T1 T value for first subsample
 * \param T2 T value for second subsample
 * 
 * \return T value for entire sample (i.e. T1+T2)
 */
inline double simpleAggregateT(double T1, double T2)
{
	return T1+T2;
}

/**
 * Aggregates two co-moment values
 * This formula comes from Chan1979, equation (5.3)
 * \param com1 Co-moment for first subsample
 * \param com2 Co-moment for second subsample
 * \param T1x T value for first feature and first subsample
 * \param T2x T value for first feature and second subsample
 * \param T1y T value for second feature and first subsample
 * \param T2y T value for second feature and second subsample
 * \param m1 Size of the first subsample
 * \param m2 Size of the second subsample
 * 
 * \return Aggregated co-moment for entire sample
 */
double simpleAggregateCom(double com1, double com2, double T1x, double T2x, double T1y, double T2y, int m1, int m2)
{
	if(m1 == 0)
		return com2;
	else if (m2 == 0)
		return com1;
	else
	{
		double ratio = (m2/(double)m1);
		return com1 + com2 + (ratio*T1x - T2x) * (ratio*T1y - T2y) / (ratio * (m1+m2));
	}
}

/**
 * Aggregates two sum-of-squares values for computing the variance
 * This formula comes from Chan1979, equation (2.1)
 * \param s1 S for first subsample
 * \param s2 S for second subsample
 * \param t1 T value for first feature and first subsample
 * \param t2 T value for first feature and second subsample
 * \param m Size of the first subsample
 * \param n Size of the second subsample
 * 
 * \return Aggregated S for entire sample
 */
double simpleAggregateS(double s1, double s2, double t1, double t2, int m, int n)
{
	if(m == 0)
		return s2;
	else if (n == 0)
		return s1;
	else
	{
		double ratio = (n/(double)m);
		return s1 + s2 + square(ratio*t1 - t2) / (ratio * (m+n));
	}
}


/**
 * Updates S for the online variance algorithm
 * \param T T value so far
 * \param S S value so far
 * \param x New value
 * \param j Number of values used to compute \p T and \p S
 * \return Updated S
 */
inline double onlineUpdateS(double T, double S, double x, int j)
{
	return S + j*square(T/j - x)/(j+1);
}

/**
 * Updates S for the online variance algorithm
 * \param Tx T value so far for first variable
 * \param Ty T value so far for second variable
 * \param com Comoment value so far
 * \param x New value for first variable
 * \param y New value for second variable
 * \param j Number of values used to compute \p Tx, \p Ty and \p com
 * \return Updated com
 */
inline double onlineUpdateCom(double Tx, double Ty, double com, double x, double y, int j)
{
	return com + j*(Tx/j - x)*(Ty/j - y)/(j+1);
}

/**
 * Splits a matrix according to the class of each element
 * 
 * \param m Input matrix
 * \param cell_classes Vector containing one element per row in \p m, indicating the class to which the corresponding row belongs
 * \param mclasses (Output) Will contain one Matrix per unique element in \p cell_classes, containing the subset of rows of \p m that belong to a certain class.
 */
void splitMatrixPerClass(const Matrix<double> &m, const vector<int> &cell_classes, vector< Matrix<double> > &mclasses)
{
	int class_counts[NUM_CLASSES];
	int class_added[NUM_CLASSES];
	
	for(int k = 0; k < NUM_CLASSES; ++k)
	{
		class_counts[k] = 0; 
		class_added[k] = 0; 
	}
	
	for(auto k = cell_classes.begin(); k != cell_classes.end(); ++k)
		++(class_counts[*k]);
	
	mclasses.resize(NUM_CLASSES);
	
	for(int k = 0; k < NUM_CLASSES; ++k)
		mclasses[k].resize(class_counts[k], m.cols());
	
	for(int i = 0; i < m.rows(); ++i)
	{
		int k = cell_classes[i];
		
		/// \todo Use memcpy
		for(int j = 0; j < m.cols(); ++j)
			mclasses[k](class_added[k], j) = m(i,j);
		
		++(class_added[k]);
	}
}


/**
 * Reads a text file that simply contains a path to a CSV file in each line, and returns the result in \p inputfiles
 * \param filelist Path of the file list
 * \param inputfiles Output parameter. Contains the list of paths read from \p filelist
 */
void readFileList(const string &filelist, vector<string> &inputfiles)
{
	ifstream fl(filelist);
	string filename;
	
	if(!fl)
	{
		cerr << "Error when trying to read filelist " << filelist << endl;
		return;
	}
	
	inputfiles.clear();
	
	fl >> filename;
	
	// Read each line of the file
	while(fl)
	{
	 inputfiles.push_back(filename);
	 fl >> filename;
	}
		
	fl.close();
}


/**
 * Reads the class names stored in a classfile and saves them in \p class_names
 * \param classfile Path of the class file
 * \param class_names Output parameter. Contains the names of the classes, the last element contains "Total" for consistency with the data structures that also gather statistics for all classes together.
 */
void getClasses(const string &classfile, vector<string> &class_names)
{
	ifstream cf(classfile);
	
	// I use a map structure to keep track of the class names
	map<string,int> classmap;
	pair<string, int> kv;
	map<string,int>::iterator ret;
	
	if(!cf)
	{
		cerr << "Error when trying to read classfile " << classfile << endl;
		return;
	}
	
	class_names.clear();
	
	cf >> kv.first;
		
	// Read each line of the file
	while(cf)
	{
		// Check if the class had been read before
		ret = classmap.find(kv.first);
		
		if(ret == classmap.end()) // New class. Insert it into the map and record the name.
		{
			kv.second = class_names.size();
			classmap.insert(kv);
			class_names.push_back(kv.first);
		}
		
		cf >> kv.first;
	}
	
	cf.close();
	
	NUM_CLASSES = class_names.size();
	
	// Add a name for the whole dataset
	class_names.push_back("Total");
}

/**
 * Merges the T and the Co-moment matrix for two parts of the dataset.
 * \param partialCom (output) Comoment (3D) of the first part of the dataset. After the call to the method, it will contain the resulting aggregated Comoment.
 * \param partialCom2 Comoment (3D) of the second part of the dataset. After the call to the method, it might contain garbage.
 * \param partialT (output) T (2D) of the first part of the dataset. After the call to the method, it will contain the resulting aggregated T.
 * \param partialT2 T (2D) of the second part of the dataset. After the call to the method, it might contain garbage.
 * \param partialDP (output) Dot product (3D) of the first part of the dataset. After the call to the method, it will contain the resulting aggregated Comoment.
 * \param partialDP2 Dot product (3D) of the second part of the dataset. After the call to the method, it might contain garbage.
 * \param count (output) Number of cells in each class, as computed by \p getClass, for the first part of the dataset. After the call to the method, it contains the total for both parts.
 * \param count2 Number of cells in each class, as computed by \p getClass, for the second part of the dataset.
 * \param m1 (Optional) Sum of \param count so that it doesn't have to be computed again.
 * \param m2 (Optional) Sum of \param count2 so that it doesn't have to be computed again.
 * \return Total number of cells considered. Corresponds to sum(\p count) after the call.
 */
int mergeTSCom(Tensor<double> &partialCom,
							 Tensor<double> &partialCom2,
							 Tensor<double> &partialT,
							 Tensor<double> &partialT2,
							 Tensor<double> &partialDP,
							 Tensor<double> &partialDP2,
							 int *count, int *count2,
							 int m1 = -1, int m2 = -1)
{
	if(m1 == -1 || m2 == -1)
	{
		m1 = m2 = 0;
		
		for(int k = 0; k < NUM_CLASSES; ++k)
		{
			m1 += count[k];
			m2 += count2[k];
		}
	}
	
	// Check if any of the parts are "empty"
	if(m1 == 0 && m2 == 0)
		return 0;
	else if(m2 == 0)
	{
		// In this case the results are already in partialT, partialCom and count
		return m1;
	}
	else if(m1 == 0)
	{
		partialT.fillValues(partialT2);
		partialCom.fillValues(partialCom2);
		partialDP.fillValues(partialDP2);
		std::swap(count, count2);
		
		return m2;
	}
	else
	{
		int num_features = partialT.dim(1);
		
		for(int k = 0; k < NUM_CLASSES; ++k)
		{
			if(count[k] == 0 && count2[k] > 0)
			{
				count[k] = count2[k];
				partialT.hyperplane(k).fillValues(partialT2.hyperplane(k));
				partialCom.hyperplane(k).fillValues(partialCom2.hyperplane(k));
				partialDP.hyperplane(k).fillValues(partialDP2.hyperplane(k));
			}
			else if(count[k] > 0 && count2[k] > 0)
			{
				for(int f = 0; f < num_features; ++f)
				{
					for(int f2 = f; f2 < num_features; ++f2)
					{
						partialCom({k, f, f2}) = simpleAggregateCom(partialCom ({k, f, f2}),
																												partialCom2({k, f, f2}),
																												partialT ({k, f }),
																												partialT2({k, f }),
																												partialT ({k, f2}),
																												partialT2({k, f2}),
																												count[k], count2[k]);
						
						partialDP({k, f, f2}) += partialDP2({k, f, f2});
					}
					
					partialT ({k, f}) = simpleAggregateT(partialT ({k, f}), partialT2 ({k, f}));
				}
				
				count[k] += count2[k];
			}
		}
		
		return m1+m2;
	}
}

/**
 * Computes the T and the Co-moment matrix for the portion of the dataset contained in \p data.
 * The results are returned into hyperplane \p first of \p Tmatrix, \p comMatrix, \p dpMatrix and \p partialM.
 * 
 * \param Tmatrix T array (3D) with the structure (subsets x NUM_CLASSES x num_features)
 * \param comMatrix Co-moment array (4D) with the structure (subsets x NUM_CLASSES x num_features x num_features)
 * \param dpMatrix Dot product array (4D) with the structure (subsets x NUM_CLASSES x num_features x num_features)
 * \param partialM 2D Matrix with the structure (subsets x NUM_CLASSES). Contains the number of cells in each class.
 * \param first First row of data to be taken into account.
 * \param last Last row of data to be taken into account.
 */
void aggregateTSComImproved(Tensor<double> &Tmatrix,
													Tensor<double> &comMatrix,
													Tensor<double> &dpMatrix,
													Matrix<int> &partialM,
													int first,
													 int last)
{
	int range = last-first+1;
	
	// Check trivial case with 1 chunk
	if(range == 1)
	{
		return;
	}
	else
	{
		int mid = last-range/2;
		
		aggregateTSComImproved(Tmatrix, comMatrix, dpMatrix, partialM, first, mid);
		aggregateTSComImproved(Tmatrix, comMatrix, dpMatrix, partialM, mid+1, last);
		
		// Pass the hyperplanes in which the results are saved to Merge function
		Hyperplane2D<double> partialTLeft  = Tmatrix  .hyperplane2D(first);
		Hyperplane2D<double> partialTRight = Tmatrix  .hyperplane2D(mid+1);
		Hyperplane<double> partialComLeft  = comMatrix.hyperplane(first);
		Hyperplane<double> partialComRight = comMatrix.hyperplane(mid+1);
		Hyperplane<double> partialDPLeft   = dpMatrix .hyperplane(first);
		Hyperplane<double> partialDPRight  = dpMatrix .hyperplane(mid+1);
		
		mergeTSCom(partialComLeft, partialComRight,
								partialTLeft  , partialTRight  ,
								partialDPLeft , partialDPRight,
						 partialM[first]  , partialM[mid+1]);
	}
}



/**
 * Merges the T and the Co-moment matrix for two parts of the dataset. The first part must be in the lower triangle and last row of the input matrices; the second part must be in the upper triangle and diagonal. The result will be stored into the former or the latter spaces depending on whether \p isleft is true or false, respectively.
 * \param partialCom (input/output) Comoment (3D) of the first part of the dataset stored in the lower triangle and last row.
 * \param partialCom2 (input/output) Comoment (3D) of the second part of the dataset stored in the upper triangle and diagonal.
 * \param partialT (input/output) T (2D) of the first part of the dataset..
 * \param partialT2 T (2D) of the second part of the dataset.
 * \param partialDP (output) Dot product (3D) of the first part of the dataset stored in the lower triangle and last row.
 * \param partialDP2 Dot product (3D) of the second part of the dataset stored in the upper triangle and diagonal.
 * \param count Number of cells in the first part.
 * \param count2 Number of cells in the second part.
 * \param isleft True if the result must be stored in the parameters of the first part, false otherwise.
 */
void mergeTSComImprovedPerClass(Hyperplane2D<double> &partialCom,
																Hyperplane2D<double> &partialCom2,
																double *partialT,
																double *partialT2,
																Hyperplane2D<double> &partialDP,
																Hyperplane2D<double> &partialDP2,
																int count,
																int count2,
																bool isleft)
{
	if(count == 0 && count2 == 0)
			return;
	else if((isleft && count2 == 0) || (!isleft && count == 0))
	{
		// In this case the results are already in their place
		return;
	}
	
	int num_features = partialCom.dim(1);
		
	if(isleft)
	{
		// The result is stored in the first part
		for(int f = 0; f < num_features; ++f)
		{
			for(int f2 = f+1; f2 < num_features; ++f2)
			{
				// The index f2 only runs over the upper triangle
				// Therefore, I swap the indexes to get the same position in the lower triangle
				partialCom(f2, f) = simpleAggregateCom(partialCom (f2, f),
																								 partialCom2(f, f2),
																								 partialT [f2],
																						 partialT2[f2],
																						 partialT [f],
																						 partialT2[f],
																										count, count2);
				
				partialDP(f2, f) += partialDP2(f, f2);
			}
			
			// Handle the diagonal and T
			partialCom(num_features, f) = simpleAggregateS(partialCom (num_features, f),
																												 partialCom2(f, f),
																												 partialT [f],
																											partialT2[f],
																														count, count2);
			
			partialDP(num_features, f) += partialDP2(f, f);
			partialT[f] = simpleAggregateT(partialT[f], partialT2[f]);
		}
	}
	else
	{
		// The result is stored in the second part
		for(int f = 0; f < num_features; ++f)
		{
			for(int f2 = f+1; f2 < num_features; ++f2)
			{
				// The index f2 only runs over the upper triangle
				// Therefore, I swap the indexes to get the same position in the lower triangle
				partialCom2(f, f2) = simpleAggregateCom(partialCom (f2, f),
																									partialCom2(f, f2),
																									partialT [f2],
																							partialT2[f2],
																							partialT [f],
																							partialT2[f],
																										 count, count2);
				
				partialDP2(f, f2) += partialDP(f2, f);
			}
			
			// Handle the diagonal and T
			partialCom2(f, f) = simpleAggregateS(partialCom (num_features, f),
																					 partialCom2(f, f),
																					 partialT [f],
																					 partialT2[f],
																					 count, count2);
			
			partialDP2(f, f) += partialDP(num_features, f);
			partialT2[f] = simpleAggregateT(partialT[f], partialT2[f]);
		}
	}
	
	// 	}
	
	// 	}
}


/**
 * Computes the T and the Co-moment matrix for the portion of the dataset contained in \p data, using the online one-pass algorithm to compute the base case.
 * \param data Portion of the dataset belonging to the same class
 * \param first First row of data to be taken into account.
 * \param last Last row of data to be taken into account.
 * \param pending_branches Number of left children of all ancestors of this node in the search tree. This number indicates the hyperplane that is available for the current iteration.
 * \param isleft True if this is the left split, false otherwise.
 * \param poolT Tensor that hosts the final and intermediate results for T, with structure (depth, 2, num_features).
 * \param poolCom Tensor that hosts the final and intermediate results for the co-moment, with structure (depth, num_features+1, num_features).
 * \param poolDP Tensor that hosts the final and intermediate results for the dot-product, with structure (depth, num_features+1, num_features).
 * \return Hyperplane of \p poolT, \p poolCom and \p poolDP in which the results are stored. If \p isleft, they will be stored in the lower triangular matrix including the diagonal; otherwise, they will be in the upper triangular matrix and the diagonal will be saved in the rightmost column.
 */
int computeTSComOnePassPerClass(Matrix<double> &data,
																 int first, int last,
																 int pending_branches,
																 bool isleft,
																 Tensor<double> &poolT,
																 Tensor<double> &poolCom,
																 Tensor<double> &poolDP)
{
	int num_features = data.cols();
	int num_cells = last-first+1;
	
	// 	cout << "Computing (" << first << "," << last << ")" <<endl;
	// Target hyperplane
	int hyp, hypT;
	
	if(isleft)
	{
		hyp = pending_branches;
		hypT = 0;
	}
	else
	{
		hyp = 0;
		hypT = 1;
	}
	
	// Check base case with few rows
	if(num_cells <= MIN_AGGREGATION_NUM)
	{
		// Where the results will be stored		
		Hyperplane2D<double> partialT   = poolT  .hyperplane2D(hyp);
		Hyperplane2D<double> partialCom = poolCom.hyperplane2D(hyp);
		Hyperplane2D<double> partialDP  = poolDP .hyperplane2D(hyp);
		
		partialT.hyperplane(hypT).fill(0.0);
		
		if(isleft)
		{
			partialCom.fillLower(0.0);
			partialDP .fillLower(0.0);
		}
		else
		{
			partialCom.fillUpperAndDiag(0.0);
			partialDP .fillUpperAndDiag(0.0);
		}
		
		// Depending on which part of the recursive tree this is,
		// fill either the lower or the upper triangle
		// The if is here for efficiency
		if(isleft)
		{
			
			// Lower triangle
			for(int f1 = num_features-1; f1 >= 0; --f1)
			{
				partialT(hypT, f1) = data(first, f1);
				partialDP(num_features, f1) = square(data(first, f1));
				for(int f2 = 0; f2 < f1; ++f2)
					partialDP (f1,f2) = data(first, f1) * data(first, f2);
			}
			
			for(int c = first+1; c <= last; ++c)
			{
				int j = c-first+1;
				
				for(int f1 = num_features-1; f1 >= 0; --f1)
				{
					partialDP (num_features,f1) += square(data(c, f1));
					partialCom (num_features,f1) = onlineUpdateS(partialT(hypT, f1),
																											 partialCom (num_features,f1),
																											 data(c, f1),
																											 j);
					
					for(int f2 = 0; f2 < f1; ++f2)
					{
						partialDP (f1,f2) += data(c, f1) * data(c, f2);
						partialCom (f1,f2) = onlineUpdateCom(partialT(hypT, f1),
																								 partialT(hypT, f2),
																								 partialCom (f1,f2),
																								 data(c, f1),
																								 data(c, f2), j);
					}
					
					partialT(hypT, f1) += data(c, f1);
				}
			}
		}
		else
		{
			// Upper triangle
			for(int f1 = 0; f1 < num_features; ++f1)
			{
				partialT(hypT, f1) = data(first, f1);
				partialDP(f1, f1) = square(data(first, f1));
				for(int f2 = f1+1; f2 < num_features; ++f2)
					partialDP (f1,f2) = data(first, f1) * data(first, f2);
			}
			
			for(int c = first+1; c <= last; ++c)
			{
				int j = c-first+1;
				
				for(int f1 = 0; f1 < num_features; ++f1)
				{
					partialDP (f1,f1) += square(data(c, f1));
					partialCom (f1,f1) = onlineUpdateS(partialT(hypT, f1),
																											 partialCom (f1,f1),
																											 data(c, f1),
																											 j);
					
					for(int f2 = f1+1; f2 < num_features; ++f2)
					{
						partialDP (f1,f2) += data(c, f1) * data(c, f2);
						partialCom (f1,f2) = onlineUpdateCom(partialT(hypT, f1),
																								 partialT(hypT, f2),
																								 partialCom (f1,f2),
																								 data(c, f1),
																								 data(c, f2), j);
					}
					
					partialT(hypT, f1) += data(c, f1);
				}
			}
		}
	}
	else
	{
		int mid = last-num_cells/2;
		
		int h1 = computeTSComOnePassPerClass(data,
																					first           , mid,
																				pending_branches  , true,
																				poolT, poolCom, poolDP);
		
		int h2 = computeTSComOnePassPerClass(data,
																					mid+1, last,
																				pending_branches+1, false,
																				poolT, poolCom, poolDP);
		
		Hyperplane2D<double> partialComLeft  = poolCom.hyperplane2D(h1);
		Hyperplane2D<double> partialComRight = poolCom.hyperplane2D(h2);
		double *           partialTLeft    = poolT  .hyperplane({h1, 0}).asVector();
		double *           partialTRight   = poolT  .hyperplane({h2, 1}).asVector();
		Hyperplane2D<double> partialDPLeft   = poolDP .hyperplane2D(h1);
		Hyperplane2D<double> partialDPRight  = poolDP .hyperplane2D(h2);
		
		mergeTSComImprovedPerClass(partialComLeft, partialComRight,
															 partialTLeft  , partialTRight,
														 partialDPLeft , partialDPRight,
														 mid-first+1,
														 last-mid,
														 isleft);
	}
	
	return hyp;
}



/**
 * Computes the T and the Co-moment matrix for the portion of the dataset contained in \p data, using the two-pass algorithm to compute the base case.
 * \param data Portion of the dataset belonging to the same class
 * \param first First row of data to be taken into account.
 * \param last Last row of data to be taken into account.
 * \param pending_branches Number of left children of all ancestors of this node in the search tree. This number indicates the hyperplane that is available for the current iteration.
 * \param isleft True if this is the left split, false otherwise.
 * \param poolT Tensor that hosts the final and intermediate results for T, with structure (depth, 2, num_features).
 * \param poolCom Tensor that hosts the final and intermediate results for the co-moment, with structure (depth, num_features+1, num_features).
 * \param poolDP Tensor that hosts the final and intermediate results for the dot-product, with structure (depth, num_features+1, num_features).
 * \return Hyperplane of \p poolT, \p poolCom and \p poolDP in which the results are stored. If \p isleft, they will be stored in the lower triangular matrix including the diagonal; otherwise, they will be in the upper triangular matrix and the diagonal will be saved in the rightmost column.
 */
int computeTSComImprovedPerClass(Matrix<double> &data,
																 int first, int last,
																 int pending_branches,
																 bool isleft,
																 Tensor<double> &poolT,
																 Tensor<double> &poolCom,
																 Tensor<double> &poolDP)
{
	int num_features = data.cols();
	int num_cells = last-first+1;
	
	// Target hyperplane
	int hyp, hypT;
	
	if(isleft)
	{
		hyp = pending_branches;
		hypT = 0;
	}
	else
	{
		hyp = 0;
		hypT = 1;
	}
	
	// Check base case with few rows
	if(num_cells <= MIN_AGGREGATION_NUM)
	{
		// Where the results will be stored		
		Hyperplane2D<double> partialT   = poolT  .hyperplane2D(hyp);
		Hyperplane2D<double> partialCom = poolCom.hyperplane2D(hyp);
		Hyperplane2D<double> partialDP  = poolDP .hyperplane2D(hyp);
		
		partialT.hyperplane(hypT).fill(0.0);
		
		if(isleft)
			partialCom.fillLower(0.0);
		else
			partialCom.fillUpperAndDiag(0.0);
		
		// For each cell compute the stats
		auto data_it = data.begin(first);
		for(int c = first; c <= last; ++c)
			for(int f = 0; f < num_features; ++f, ++data_it)
				partialT(hypT, f) += *data_it;
		
		// Compute average
		double averages[num_features];
		
		for(int f = 0; f < num_features; ++f)
			averages[f] = partialT(hypT, f) / num_cells;
		
		// For efficiency, we first substract the average from the values.
		// This also improves the numerical robustness
		data_it = data.begin(first);
		for(int c = first; c <= last; ++c)
			for(int f = 0; f < num_features; ++f, ++data_it)
				*data_it -= averages[f];
		
		
		// Depending on which part of the recursive tree this is,
		// fill either the lower or the upper triangle
		if(isleft)
		{
			// Lower triangle
			for(int c = first; c <= last; ++c)
			{
				for(int f1 = 0; f1 < num_features; ++f1)
				{
					double &tmp = data(c, f1);
					partialCom(num_features,f1) += square(tmp);
					
					for(int f2 = 0; f2 < f1; ++f2)
						partialCom(f1,f2) += tmp * data(c, f2);
				}
			}
			
			for(int f1 = 0; f1 < num_features; ++f1)
			{
				double tmp = num_cells*averages[f1];
				
				partialDP(num_features,f1) = partialCom(num_features,f1) + tmp*averages[f1];
				
				for(int f2 = 0; f2 < f1; ++f2)
					partialDP(f1,f2) = partialCom(f1,f2) + tmp*averages[f2];
			}
			
		}
		else
		{
			// Upper triangle
			for(int c = first; c <= last; ++c)
			{
				for(int f1 = 0; f1 < num_features; ++f1)
				{
					double &tmp = data(c, f1);
					partialCom (f1,f1) += square(tmp);
					
					for(int f2 = f1+1; f2 < num_features; ++f2)
						partialCom (f1,f2) += tmp * data(c, f2);
				}
			}
			
			for(int f1 = 0; f1 < num_features; ++f1)
			{
				double tmp = num_cells*averages[f1];
				
				partialDP(f1,f1) = partialCom(f1,f1) + tmp*averages[f1];
				
				for(int f2 = f1+1; f2 < num_features; ++f2)
					partialDP(f1,f2) = partialCom(f1,f2) + tmp*averages[f2];
			}
		}
	}
	else
	{
		int mid = last-num_cells/2;
		int h1 = computeTSComImprovedPerClass(data,
																					first           , mid,
																				pending_branches  , true,
																				poolT, poolCom, poolDP);
		int h2 = computeTSComImprovedPerClass(data,
																					mid+1, last,
																				pending_branches+1, false,
																				poolT, poolCom, poolDP);
		
		Hyperplane2D<double> partialComLeft  = poolCom.hyperplane2D(h1);
		Hyperplane2D<double> partialComRight = poolCom.hyperplane2D(h2);
		double *           partialTLeft    = poolT  .hyperplane({h1, 0}).asVector();
		double *           partialTRight   = poolT  .hyperplane({h2, 1}).asVector();
		Hyperplane2D<double> partialDPLeft   = poolDP .hyperplane2D(h1);
		Hyperplane2D<double> partialDPRight  = poolDP .hyperplane2D(h2);
		
		mergeTSComImprovedPerClass(partialComLeft, partialComRight,
															 partialTLeft  , partialTRight,
														 partialDPLeft , partialDPRight,
														 mid-first+1,
														 last-mid,
														 isleft);
	}
	
	return hyp;
}





/**
 * Computes the T and the Co-moment matrix for the portion of the dataset contained in \p data. This function is improved to minimize the use of memory for intermediate computations, which becomes O(log N)
 * \param data Portion of the dataset. Each element of the vector corresponds to a different class.
 * \param partialT (output) Computed T (2D). Each row is a class, each column a feature.
 * \param partialCom (output) Computed Comoment (3D). The first dimension corresponds to the class, second and third dimensions conform the comoment matrix for each class.
 * \param partialDP (output) Computed dot product (3D). The first dimension corresponds to the class, second and third dimensions conform the comoment matrix for each class.
 * \param count (output) Number of cells (i.e. rows of data) in each class, as computed by \p getClass.
 */
void computeTSComImprovedPerClass(vector< Matrix<double> > &data,
													Tensor<double> &partialT,
													Tensor<double> &partialCom,
													Tensor<double> &partialDP,
													int *count)
{
	int num_features = partialT.dim(1);
	
	// Computing the maximum depth of the recursive calls
	int maxrows = 0;
	
	for(unsigned int k = 0; k < data.size(); ++k)
	{
		count[k] = data[k].rows();
		if(count[k] > maxrows)
			maxrows = count[k];
		
// 		data[k].transpose();
	}
	
	int m0 = std::min(MIN_AGGREGATION_NUM, maxrows);
	int depth = int(ceil(log2(maxrows / m0 + 1)));
	
	// First approach. I reserve an independent structure for the recursive computation, and then I copy it.
	Tensor<double> poolT  ({depth, 2, num_features});
	Tensor<double> poolCom({depth, num_features+1, num_features});
	Tensor<double> poolDP ({depth, num_features+1, num_features});
	
	for(unsigned int k = 0; k < data.size(); ++k)
	{
		if(count[k] > 0)
		{
			computeTSComImprovedPerClass(data[k], 0, count[k]-1,
																	 0, false, poolT, poolCom, poolDP);
			partialT  .hyperplane(k).fillValues(poolT  .hyperplane({0, 1}));
			partialCom.hyperplane(k).fillValues(poolCom.hyperplane(0));
			partialDP .hyperplane(k).fillValues(poolDP .hyperplane(0));
		}
	}
}


/**
 * Checks whether the current process is the master
 * \return True if the current process is the master
 */
bool isMaster()
{
	int current = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &current);
	return (current == MASTER_PROCESS);
}


/**
 * Gathers a matrix \p msend from all processes using MPI, and the master process returns the result in Tensor \p mresult.
 * \param msend Matrix that will be sent from all processes.
 * \param mresult Gathered 3D array (only for Master process).
 * \param type MPI data type
 */
template<class T>
void gatherMatrix(const Matrix<T> &msend, Tensor<T> &mresult, MPI_Datatype type)
{
	int count = msend.rows() * msend.cols();
	
	if(isMaster())
	{
		int num_procs = 0;
		MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
		mresult.reserve({num_procs, msend.rows(), msend.cols()});
	}
	
	MPI_Gather(msend.asVector(), count, type, mresult.asVector(), count, type, MASTER_PROCESS, MPI_COMM_WORLD);
}

/**
 * Gathers a vector \p msend from all processes using MPI, and the master process returns the result in a matrix \p mresult.
 * \param msend Vector that will be sent from all processes.
 * \param count Number of elements in send buffer.
 * \param mresult Gathered matrix (only for Master process).
 * \param type MPI data type
 */
template<class T>
void gatherMatrix(const T *msend, int count, Matrix<T> &mresult, MPI_Datatype type)
{
	if(isMaster())
	{
		int num_procs = 0;
		MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
		mresult.resize(num_procs, count);
	}
	
	MPI_Gather(msend, count, type, mresult.asVector(), count, type, MASTER_PROCESS, MPI_COMM_WORLD);
}


/**
 * Gathers a Tensor \p msend from all processes using MPI, and the master process returns the result in a Tensor \p mresult.
 * \param msend N-dimensional Tensor that will be sent from all processes.
 * \param mresult N+1-dimensional gathered Tensor (only for Master process).
 * \param type MPI data type
 */
template<class T>
void gatherTensor(const Tensor<T> &msend, Tensor<T> &mresult, MPI_Datatype type)
{
	int count = msend.size();
	
	if(isMaster())
	{
		int num_procs = 0;
		MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
		
		vector<int> dims(msend.dimnum()+1);
		
		MPI_Comm_size(MPI_COMM_WORLD, &(dims[0]));
		
		for(int i = 0; i < msend.dimnum(); ++i)
			dims[i+1] = msend.dim(i);
		
		mresult.reserve(dims);
	}
	
	MPI_Gather(msend.asVector(), count, type, mresult.asVector(), count, type, MASTER_PROCESS, MPI_COMM_WORLD);
}



/**
 * Reads the first line of a CSV file and returns a vector containing the names of the columns, and their number
 * \param mfile Input stream pointing to the CSV file
 * \param feature_names Output parameter. Contains the names of the read columns.
 * \return The number of features (i.e. \p feature_names.size())
 */
int getFeaturesCSVStream(istream &mfile, vector<string> &feature_names)
{
	int num_features = 0;
	string buffer;
	
	// Read first line
	mfile >> buffer;
	
	num_features = std::count(buffer.begin(), buffer.end(), ',');
	
	if(NUM_CLASSES == 1)
		num_features++;
	
	cout << "The master process detected " << num_features << " features" << endl;
	
	stringstream ss(buffer);
	ss.seekg(0, std::ios::beg);
	string token;
	
	for(int j = 0; j < num_features; ++j)
	{
		std::getline(ss, token, ',');
		feature_names.push_back(token);
	}
	
	return num_features;
}


/**
 * Reads the first line of a CSV file and returns a vector containing the names of the columns, and their number
 * \param filename The name of the CSV file
 * \param feature_names Output parameter. Contains the names of the read columns.
 * \return The number of features (i.e. \p feature_names.size())
 */
int getFeaturesCSV(const string &filename, vector<string> &feature_names)
{
	int res = 0;
	
	if(DEBUG)
		cout << "The master process is counting the features in file " << filename << endl;
	
	// If the file is gzipped, convert it first
	if(filename.find(".gz") == filename.size()-3)
	{
		ifstream file(filename.c_str(), ios_base::in | ios_base::binary);
		boost::iostreams::filtering_streambuf<boost::iostreams::input> inbuf;
		inbuf.push(boost::iostreams::gzip_decompressor());
		inbuf.push(file);
		
		//Convert streambuf to istream
		istream mfile(&inbuf);
		
		if(!mfile)
			exit(string("Error when reading file ") + filename, -1);
		
		res = getFeaturesCSVStream(mfile, feature_names);
		file.close();
	}
	else
	{
		ifstream mfile(filename.c_str());
		
		if(!mfile)
			exit(string("Error when reading file ") + filename, -1);
		
		res = getFeaturesCSVStream(mfile, feature_names);
		mfile.close();
	}
	
	return res;
}



/**
 * Main loop
 * \param argc
 * \param argv mpirun -np \<processes\> executable {--classfile \<classfile.txt\> | --numclasses \<num\>} [--outputdir \<path\>] [--debug] \<file1.csv\> ...
 * \return Exit value
 */    
int main(int argc, char *argv[])
{
	int provided = 0;
	
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
	
	if(provided < MPI_THREAD_FUNNELED)
		cerr << "WARNING: the level of parallelism is lower than MPI_THREAD_FUNNELED" << endl;
	
	// Automatically use one thread per core
	omp_set_num_threads(omp_get_num_procs());

	if(DEBUG)
	{
		#pragma omp critical
		cout << "Number of threads set to " << omp_get_max_threads() << endl;
	}
	
	if(argc < 4)
	{
		exit("Syntax error: mpirun -np <processes> executable [--classfile <classfile.txt> | --numclasses <num>] [--outputdir <path>] [--debug] [--times] {--filelist <list.txt> | <file1.csv> ...}", -1);
	}
	
	string classfile = "";
	string output_dir = ".";
	string filelist = "";
	vector<string> inputfiles;
	vector<string> class_names;
	TimeLog timelog;
	
	// Read the parameters
	for(int i = 1; i < argc; ++i)
	{
		if(strcmp(argv[i], "--classfile") == 0)
		{
			if(NUM_CLASSES > 0)
				exit("At most one of --classfile and --numclasses should be provided", -2);
			
			classfile = argv[i+1];
			++i;
			
			getClasses(classfile, class_names);
		}
		else if(strcmp(argv[i], "--numclasses") == 0)
		{
			if(NUM_CLASSES > 0)
				exit("At most one of --classfile and --numclasses should be provided", -2);
			
			NUM_CLASSES = atoi(argv[i+1]);
			++i;
		}
		else if(strcmp(argv[i], "--minaggregation") == 0)
		{
			MIN_AGGREGATION_NUM = atoi(argv[i+1]);
			++i;
		}
		else if(strcmp(argv[i], "--outputdir") == 0)
		{			
			output_dir = string(argv[i+1]);
			++i;
		}
		else if(strcmp(argv[i], "--filelist") == 0)
		{			
			filelist = string(argv[i+1]);
			++i;
			
			readFileList(filelist, inputfiles);
		}
		else if(strcmp(argv[i], "--debug") == 0)
			DEBUG = true;
		else if(strcmp(argv[i], "--times") == 0)
			MEASURE_TIMES = true;
		else
		{
			if(filelist != "")
				exit("If --filelist is provided, there should be no trailing arguments", -3);
			
			inputfiles.push_back(argv[i]);	
		}
	}
	
	if(NUM_CLASSES <= 0)
	{
		NUM_CLASSES = 1;
		class_names.push_back("Total");
	}
	
	int num_files = inputfiles.size();
	int num_procs = 0;
	int current_process = 0;
	
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &current_process);
	
	if(current_process == MASTER_PROCESS)
		cout << "Read a total of " << inputfiles.size() << " files" << endl;
	
	if(num_files < num_procs)
		exit("ERROR: there are less files than processes", -1);
	
	// Each process computes which files it will take care of
	int begin, end;
	int chunk_size = (int)floor(num_files/num_procs);
	
	// We want consecutive chunks, as equally distributed as possible
	begin = chunk_size*current_process + fmin((num_files % num_procs), current_process);
	end = begin + chunk_size - 1;
	
	if(current_process < num_files % num_procs)
		end++;
	
	if(DEBUG)
		cout << "Process " << current_process << " will read: [" << begin << "," << end << "]" << endl;
	
	vector<string> feature_names;
	int num_features = 0;
	
	// Read the number of features
	if(isMaster())
			num_features = getFeaturesCSV(inputfiles[0], feature_names);
	
	if(MEASURE_TIMES)
		timelog.addTimeStamp("Features read");
	
	if(DEBUG && isMaster())
		cout << "Broadcasting number of features: " << num_features << endl;
	
	MPI_Bcast(&num_features,  1,  MPI_INT,  MASTER_PROCESS, MPI_COMM_WORLD);
	
	if(MEASURE_TIMES)
		timelog.addTimeStamp("Features broadcasted");
	if(DEBUG)
		cout << "Process " << current_process << " received number of features: " << num_features << endl;

	// Reserve space for the T of each feature, each class and each input file
	Tensor<double> T{end-begin+1, NUM_CLASSES, num_features};
	T.fill(0.0);
	
	// Reserve space for the Comoment matrices (4 dimensions)
	Tensor<double> threadComs{end-begin+1, NUM_CLASSES, num_features, num_features};
	
	// Reserve space for the Dot Product matrices (4 dimensions)
	Tensor<double> threadDP{end-begin+1, NUM_CLASSES, num_features, num_features};
	
	Matrix<int> partialM(end-begin+1, NUM_CLASSES);
	
	if(MEASURE_TIMES)
		timelog.addSyncTimeStamp("Structures created");
	
	if(DEBUG)
		cout << "Process " << current_process << " created the structures" << endl;
	
	double time_io = 0, time_basecases = 0;
	
	// Read list of csv files
	// Each process reads its csv files (in parallel)
	#pragma omp parallel for shared(T, threadComs, threadDP, partialM) reduction(+ : time_io, time_basecases)
	for(int i = begin; i <= end; ++i)
	{
		vector< Matrix<double> > mclasses;
		
		if(DEBUG)
		{
			#pragma omp critical
			cout << "Process " << current_process << " reading file " << inputfiles[i] << endl;
		}
		
		// Let's control the time of the I/O and that of the base cases separately
		double time_begin = MPI_Wtime();
		
		if(NUM_CLASSES > 1)
		{
			Matrix<double> m;
			vector<int> cell_classes;
			
			m.readFromCSV(inputfiles[i], class_names, cell_classes, true, num_features);	
			
			splitMatrixPerClass(m, cell_classes, mclasses);
		}
		else
		{
			mclasses.resize(1);
			mclasses[0].readFromCSV(inputfiles[i], true, num_features);
		}
		
		time_io = MPI_Wtime() - time_begin;
		
 		if(DEBUG)
		{
			#pragma omp critical
			cout << "Process " << current_process << " just read file " << inputfiles[i] << endl;	
		}
		
		time_begin = MPI_Wtime();

		// Each slave computes the T and S for all features in the current file
		Hyperplane<double> Thyp   = T         .hyperplane(i-begin);
		Hyperplane<double> Comhyp = threadComs.hyperplane(i-begin);
		Hyperplane<double> DPhyp  = threadDP  .hyperplane(i-begin);
		
		computeTSComImprovedPerClass(mclasses, Thyp, Comhyp, DPhyp, partialM[i-begin]);
		
		time_basecases = MPI_Wtime() - time_begin;
		
		if(DEBUG)
		{
			#pragma omp critical
			cout << "Process " << current_process << " ended the computation of file " << inputfiles[i] << endl;
		}
	}
	
	if(MEASURE_TIMES)
	{
		double sum_io = 0;
		double sum_basecases = 0;
		
		timelog.addSyncTimeStamp("CSV files read and base cases computed");
		
		MPI_Reduce(&time_io, &sum_io, 1, MPI_DOUBLE, MPI_SUM, MASTER_PROCESS, MPI_COMM_WORLD);
		MPI_Reduce(&time_basecases, &sum_basecases, 1, MPI_DOUBLE, MPI_SUM, MASTER_PROCESS, MPI_COMM_WORLD);
		
		if(isMaster())
		{
			double auxtimestamp = timelog.getLastValue();
			
			timelog.addManualTimeStamp("CSV files read (total)", sum_io);
			timelog.addManualTimeStamp("Base cases computed (total)", sum_basecases);
			
			double ratio = sum_io / (sum_io + sum_basecases);
			
			timelog.addManualTimeStamp("CSV files read (corrected)", auxtimestamp*ratio);
			timelog.addManualTimeStamp("Base cases computed (corrected)", auxtimestamp*(1-ratio));
			
			timelog.reset();
		}
	}
	
	if(DEBUG)
		cout << "Process " << current_process << " will aggregate its partial T and Com" << endl;
	
	// Each slave aggregates its partial T and S
	aggregateTSComImproved(T, threadComs, threadDP, partialM, 0, end-begin);
	
	int total_cells = 0;
	for(int i = 0; i < NUM_CLASSES; ++i)
		total_cells += partialM(0, i);
	
	cout << "Process " << current_process << " processed " << total_cells << " cells" << endl;
	
	if(MEASURE_TIMES)
		timelog.addSyncTimeStamp("Values aggregated");
	
	// The master gathers the partial T and Com
	Matrix<int> process_class_cells;
	Tensor<double> processT;
	Tensor<double> processCom;
	Tensor<double> processDP;
	
	// Gather the number of cells of each class processed by each process
	if(DEBUG)
		cout << "Process " << current_process << " gathering results" << endl;
	
	// Gathering all results
	gatherMatrix(partialM[0], NUM_CLASSES, process_class_cells, MPI_INT);
	gatherTensor(T.hyperplane(0), processT, MPI_DOUBLE);
	gatherTensor(threadComs.hyperplane(0), processCom, MPI_DOUBLE);
	gatherTensor(threadDP.hyperplane(0), processDP, MPI_DOUBLE);
	
	if(MEASURE_TIMES)
		timelog.addSyncTimeStamp("All values gathered by the master");
	
	if(current_process == MASTER_PROCESS)
		cout << "All values gathered by the master" << endl;
	
	// Aggregation (recursive)
	if(current_process == MASTER_PROCESS)
	{
		int mclass_cells[NUM_CLASSES+1];
		
		// Aggregate the gathered results
		aggregateTSComImproved(processT, processCom, processDP, process_class_cells, 0, num_procs-1);
		
		// We need one more hyperplane to aggregate the results across classes
		/// \todo This might be more efficient if I just save it in a different structure, so I avoid copying the entire tensors?		
		Tensor<double> finalT  {NUM_CLASSES+1, num_features};
		Tensor<double> finalCom{NUM_CLASSES+1, num_features, num_features};
		Tensor<double> finalDP {NUM_CLASSES+1, num_features, num_features};
		
		finalT  .fillValues(processT  .hyperplane(0));
		finalCom.fillValues(processCom.hyperplane(0));
		finalDP .fillValues(processDP .hyperplane(0));
		
		for(int k = 0; k < NUM_CLASSES; ++k)
			mclass_cells[k] = process_class_cells(0, k);
		
		// Free some memory
		processT.clear();
		processCom.clear();
		processDP.clear();
		
		if(NUM_CLASSES > 1)
		{
			finalT.hyperplane(NUM_CLASSES).fill(0.0);
			finalCom.hyperplane(NUM_CLASSES).fill(0.0);
			finalDP.hyperplane(NUM_CLASSES).fill(0.0);
			mclass_cells[NUM_CLASSES] = 0;
			
			// Compute the stats for all the classes together
			for(int k = 0; k < NUM_CLASSES; ++k)
				if(mclass_cells[k] > 0)
				{
					for(int i = 0; i < num_features; ++i)
					{
						for(int j = i; j < num_features; ++j)
						{
							finalCom({NUM_CLASSES, i, j}) = simpleAggregateCom(
								finalCom({NUM_CLASSES, i, j}),
																																 finalCom({k, i, j}),
																																 finalT({NUM_CLASSES,i}),
																																 finalT({k,i}),
																																 finalT({NUM_CLASSES,j}),
																																 finalT({k,j}),
																																 mclass_cells[NUM_CLASSES],
																													mclass_cells[k]
							);
							
							finalDP({NUM_CLASSES, i, j}) = finalDP({NUM_CLASSES, i, j}) + finalDP({k, i, j});
						}
					}
					
					// Update T for the cummulated set of classes, for every feature
					for(int i = 0; i < num_features; ++i)
						finalT({NUM_CLASSES, i}) = simpleAggregateT(finalT({NUM_CLASSES, i}),
																												finalT({k, i}));
						
					mclass_cells[NUM_CLASSES] += mclass_cells[k];
				}
		}
		
		if(MEASURE_TIMES)
			timelog.addTimeStamp("Values aggregated by the master");
		
		// Quick and dirty fix for the case when we have a single class:
		if(NUM_CLASSES == 1)
			NUM_CLASSES = 0;
		
		cout << "Total number of cells: " << mclass_cells[NUM_CLASSES] << endl;
			
		// Compute the distaneces from the covariance matrix
		Tensor<double> cor_pearson  {NUM_CLASSES+1, num_features, num_features};
		Tensor<double> covariance   {NUM_CLASSES+1, num_features, num_features};
		Tensor<double> dist_pearson {NUM_CLASSES+1, num_features, num_features};
		Tensor<double> dist_pearson2{NUM_CLASSES+1, num_features, num_features};
		Tensor<double> mici         {NUM_CLASSES+1, num_features, num_features};
		Tensor<double> cosine       {NUM_CLASSES+1, num_features, num_features};
		
		Matrix<double> stdev(NUM_CLASSES+1, num_features);
		Matrix<double> averages(num_features, NUM_CLASSES+1);
		
		// Compute the covariance and the stdev
 		#pragma omp parallel for
		for(int k = 0; k <= NUM_CLASSES; ++k)
			if(mclass_cells[k] > 0)
			{
				if(DEBUG)
					cout << "Cells in class " << k << ": " << mclass_cells[k] << endl;
				
				#pragma omp parallel for
				for(int i = 0; i < num_features; ++i)
				{
					covariance({k,i,i}) = finalCom({k,i,i}) / (mclass_cells[k]-1);
					averages(i, k) = finalT({k, i}) / mclass_cells[k];
					
					for(int j = i+1; j < num_features; ++j)
					{
						covariance({k,i,j}) = finalCom({k,i,j}) / (mclass_cells[k]-1);
						covariance({k,j,i}) = covariance({k,i,j});
					}
					
					// Compute the std. deviation
					stdev(k, i) = sqrt(covariance({k,i,i}));
				}
			}
		
		// Now compute Pearson's correlation and the MICI
		#pragma omp parallel for
		for(int k = 0; k <= NUM_CLASSES; ++k)
		{
			if(mclass_cells[k] > 0)
			{
				#pragma omp parallel for
				for(int i = 0; i < num_features; ++i)
				{
					cor_pearson({k,i,i}) = 1;
					mici({k,i,i}) = 0;
					dist_pearson({k,i,i}) = 0;
					dist_pearson2({k,i,i}) = 0;
					cosine({k,i,i}) = 0;
					
					double covii = covariance({k,i,i});
					
					for(int j = i+1; j < num_features; ++j)
					{
						cor_pearson({k,i,j}) = fmin(fmax(covariance({k,i,j}) / (stdev(k,i)*stdev(k,j)), -1), 1);
						cor_pearson({k,j,i}) = cor_pearson({k,i,j});
						
						dist_pearson({k,i,j}) = 1-cor_pearson({k,i,j});
						dist_pearson({k,j,i}) = dist_pearson({k,i,j});
						
						dist_pearson2({k,i,j}) = 1-abs(cor_pearson({k,i,j}));
						dist_pearson2({k,j,i}) = dist_pearson2({k,i,j});
						
						mici({k,i,j}) = 0.5*fmax(covii + covariance({k,j,j}) - sqrt( square(covii - covariance({k,j,j})) + 4*square(covariance({k,i,j})) ), 0);
						mici({k,j,i}) = mici({k,i,j});
						
						// Using sqrt twice is less efficient but less prone to overflow
						cosine({k,i,j}) = 1. - ( finalDP({k,i,j}) / (sqrt(finalDP({k,i,i})) * sqrt(finalDP({k,j,j}))) );
						cosine({k,j,i}) = cosine({k,i,j});
					}
				}
				
				// Save files
				Matrix<double> cor_pearson_matrix;
				Matrix<double> dist_pearson_matrix;
				Matrix<double> dist_pearson2_matrix;
				Matrix<double> mici_matrix;
				Matrix<double> covariance_matrix;
				Matrix<double> cosine_matrix;
				Matrix<double> dotproduct_matrix;
				
				cor_pearson.newFromHyperplane(cor_pearson_matrix, k);
				dist_pearson.newFromHyperplane(dist_pearson_matrix, k);
				dist_pearson2.newFromHyperplane(dist_pearson2_matrix, k);
				mici.newFromHyperplane(mici_matrix, k);
				covariance.newFromHyperplane(covariance_matrix, k);
				cosine.newFromHyperplane(cosine_matrix, k);
				finalDP.newFromHyperplane(dotproduct_matrix, k);
				
				string current_class = class_names[k];
				
				cor_pearson_matrix  .writeCSVwHeaders(output_dir + "/cor_pearson_" + current_class + ".csv", feature_names, feature_names);
				dist_pearson_matrix .writeCSVwHeaders(output_dir + "/pearson_"     + current_class + ".csv", feature_names, feature_names);
				dist_pearson2_matrix.writeCSVwHeaders(output_dir + "/pearson2_"    + current_class + ".csv", feature_names, feature_names);
				mici_matrix         .writeCSVwHeaders(output_dir + "/mici_"        + current_class + ".csv", feature_names, feature_names);
				covariance_matrix   .writeCSVwHeaders(output_dir + "/covariance_"  + current_class + ".csv", feature_names, feature_names);
				cosine_matrix       .writeCSVwHeaders(output_dir + "/cosine_"      + current_class + ".csv", feature_names, feature_names);
				dotproduct_matrix   .writeCSVwHeaders(output_dir + "/dotproduct_"  + current_class + ".csv", feature_names, feature_names);
			}
		}
		
		if(MEASURE_TIMES)
			timelog.addTimeStamp("Values saved by the master");
		
		averages.writeCSVwHeaders(output_dir + "/averages.csv",
															feature_names,
															vector<string>(std::begin(class_names), std::end(class_names)));
		
		if(MEASURE_TIMES)
			timelog.saveCSV(output_dir + "/timelog.csv");
		
		// Write total and averages to file
		std::ofstream mfile(output_dir + "/counts.csv");
		
		if(!mfile)
		{
			std::cerr << "Main: Error when opening file " << output_dir << "/counts.csv for output " << std::endl;
		}
		else
		{
// 			mfile << "Total number of cells: " << total_cells << endl << endl;
// 			mfile << "Count by class:" << endl;
			if(NUM_CLASSES > 0)
			{
				mfile << "Class,Count,Probability" << endl;
				for(int i = 0; i < NUM_CLASSES; ++i)
					mfile << class_names[i] << "," << mclass_cells[i] << "," << mclass_cells[i]/((double) total_cells) << endl;
				
				mfile.close();
			}
		}
	}
	
	MPI_Finalize();
}

