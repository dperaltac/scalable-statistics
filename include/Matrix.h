/**
 * \file    Matrix.h
 * \author  Daniel Peralta <daniel.peralta@irc.vib-ugent.be>
 * \version 2.0
 *
 * \section DESCRIPTION_MATRIX
 *
 * A template for building generic matrixes
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <limits>
#include <iostream>
#include <list>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <vector>

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include "Functions.h"

#include "mpi.h"

template<typename T> class Matrix;
template<typename T> std::ostream& operator<<(std::ostream& out, const Matrix<T>& M);

template <typename T>

/**
 * @class Matrix
 *
 * The Matrix class allows to define generic matrixes, able to store any of the primitive types.
 * It is meant to be more efficient than dynamically reserved vectors of vectors.
 * It supports also some additional functions, such as filling or printing (<<).
 * \todo Maybe turn this into a specialization of Tensor
 */
class Matrix {

public:
	
	
	// iterator
	typedef T* iterator;	///< Iterator
	typedef const T* const_iterator;	///< Constant iterator
	
	// iterator functions
	/**
	 * Iterator pointing to the first element of the matrix
	 * \return Iterator pointing to the first element of the matrix
	 */
	inline iterator begin() { return contents[0]; }
	
	/**
	 * Iterator pointing to the end of the matrix
	 * \return Iterator pointing to the end of the matrix
	 */
	inline iterator end()   { return contents[0] + size(); }
	
	/**
	 * Iterator pointing to the first element of the matrix
	 * \return Iterator pointing to the first element of the matrix
	 */
	inline const_iterator begin() const { return contents[0]; }
	
	/**
	 * Iterator pointing to the end of the matrix
	 * \return Iterator pointing to the end of the matrix
	 */
	inline const_iterator end() const   { return contents[0] + size(); }
	
	/**
	 * Iterator pointing to the beginining of row \p row
	 * \param row Selected row
	 * \return Iterator pointing to row \p row
	 */
	inline iterator begin(int row) { return contents[row]; }
	
	/**
	 * Iterator pointing to the end of row \p row
	 * \param row Selected row
	 * \return Iterator pointing to row \p row
	 */
	inline iterator end(int row)   { return contents[row+1]; }
	
	/**
	 * Iterator pointing to the beginining of row \p row
	 * \param row Selected row
	 * \return Iterator pointing to row \p row
	 */
	inline const_iterator begin(int row) const { return contents[row]; }
	
	/**
	 * Iterator pointing to the end of row \p row
	 * \param row Selected row
	 * \return Iterator pointing to row \p row
	 */
	inline const_iterator end(int row) const   { return contents[row+1]; }

    /** Default Constructor */
	Matrix();

	/**
	 * Constructor
	 * \param rows Rows of the matrix
	 * \param cols Columns of the matrix
	 */
	Matrix(int rows, int cols);

	/**
	 * Constructor
	 * \param rows Rows of the matrix
	 * \param cols Columns of the matrix
	 * \param value Initial value
	 */
	Matrix(int rows, int cols, const T &value);

	/**
	 * Constructor. Builds a matrix out from a vector, reusing the memory.
	 * \param v Vector
	 * \param rows Rows of the matrix
	 * \param cols Columns of the matrix
	 */
	Matrix(T *v, int rows, int cols);

    /** Default destructor */
    ~Matrix();

    /**
     * Copy constructor
     * \param o Object to copy from
     */
    Matrix(const Matrix& o);

    /**
     * Assignment operator
     * \param o Object to assign from
     * \return A reference to this object
     */
    Matrix & operator= (const Matrix &o);

    /**
     * Erase all the contents of the matrix
     */
    void clear();

    /**
     * Changes the size of the matrix.
     * \param rows New number of rows of the matrix
     * \param columns New number of columns of the matrix
     */
    void resize(int rows, int columns);

    /**
     * Access operator
     * \param i Row index
     * \param j Column index
     * \return A reference the value stored in (i,j)
     */
    const T& operator() (int i, int j) const;

    /**
     * Get vector operator
     * \param i Row index
     * \return A pointer to the i-th row
     */
    const T* operator[] (int i) const;

    /**
     * Access operator
     * \param i Row index
     * \param j Column index
     * \return A reference the value stored in (i,j)
     */
    T& operator() (int i, int j);

    /**
     * Get vector operator
     * \param i Row index
     * \return A pointer to the i-th row
     */
    T* operator[] (int i);

    /**
     * Get the number of rows of the matrix
     * \return Number of rows of the matrix
     */
    int rows() const;

    /**
     * Get the number of columns of the matrix
     * \return Number of columns of the matrix
     */
    int cols() const;

    /**
     * Get the number of elements of the matrix (rows*columns)
     * \return Number of elements of the matrix
     */
    int size() const;

    /**
     * Get if the matrix is empty or not
     * \return true if the matrix if empty, false otherwise
     */
    bool empty() const;

    /**
     * Get the minimum dimension of the matrix
     * \return the minimum dimension of the matrix
     */
    int minsize() {
        return ((nRows < nCols) ? nRows : nCols);
    }

    /**
     * Fills the matrix with a given value
     * \param value Value used for filling the matrix
     */
    void fill(const T &value);

    /**
		 * Output operator
		 */
		friend std::ostream& operator<< <>(std::ostream& out, const Matrix<T>& M);
		
		/**
		 * Return the pointer of the matrix
		 * \return Pointer to the start of the matrix
		 */
		T** getPointer();
		
		/**
		 * Return the pointer of the matrix, from where all the data can be read as a single vector.
		 * \return Pointer to the start of the matrix
		 */
		T* asVector();
		
		/**
		 * Efficiently swaps the contents of the matrix by that of \p o
		 * \param o Matrix to be swapped by \p this
		 */
		void swap(Matrix &o);
		
		/**
		 * Reads a matrix from a csv, where the last column is the class, read as either a character string of as integers (mapping to classes in \p class_names).
		 * \param mfile Input stream pointint to the file
		 * \param class_names Vector containing the class names. Especially suited to R factors.
		 * \param classes (output) When the method finishes, contains the class of each instance
		 * \param header Whether the file has column headers
		 * \param numcols Number of columns (without including the class), if already known it will improve the efficiency of the method.
		 * \param numlines Number of rows (without including the header), if already known it will improve the efficiency of the method.
		 * \todo Include a parameter to change the position of the class column, or detect it from the names.
		 */
		void readFromCSV(std::istream &mfile, const std::vector<std::string> &class_names, std::vector<int> &classes, bool header = true, int numcols = -1, int numlines = -1);
		
		/**
		 * Reads a matrix from a CSV file, which can be gzipped
		 * \param filename Name of the CSV file
		 * \param header Whether the file has column headers not
		 * \param numcols Number of columns, if already known it will improve the efficiency of the method.
		 */
		void readFromCSV(const std::string &filename, bool header = true, int numcols = -1);
		
		/**
		 * Reads a matrix from a csv, where the last column is the class, read as either a character string of as integers (mapping to classes in \p class_names).
		 * \param fcontents Contents of the file
		 * \param fsize Number of characters in the file
		 * \param class_names Vector containing the class names. Especially suited to R factors.
		 * \param classes (output) When the method finishes, contains the class of each instance
		 * \param header Whether the file has column headers
		 * \param numcols Number of columns (without including the class), if already known it will improve the efficiency of the method.
		 * \todo Include a parameter to change the position of the class column, or detect it from the names.
		 */
		void readFromCSV(char *fcontents, long fsize, const std::vector<std::string> &class_names, std::vector<int> &classes, bool header = true, int numcols = -1);
		
		/**
		 * Reads a matrix from a csv, where the last column is the class, read as either a character string of as integers (mapping to classes in \p class_names).
		 * \param filename Name of the csv file
		 * \param class_names Vector containing the class names. Especially suited to R factors.
		 * \param classes (output) When the method finishes, contains the class of each instance
		 * \param header Whether the file has column headers
		 * \param numcols Number of columns (without including the class), it already known it will improve the efficiency of the method.
		 * \todo Include a parameter to change the position of the class column, or detect it from the names.
		 */
		void readFromCSV(const std::string &filename, const std::vector<std::string> &class_names, std::vector<int> &classes, bool header = true, int numcols = -1);
		
		/**
		 * Reads a matrix from a csv
		 * \param mfile File descriptor
		 */
		void readFromCSVRowLimitStatic(std::ifstream &mfile);
		
		/**
		 * Reads a matrix from a csv
		 * \param mfile File descriptor
		 */
		void readFromCSVRowLimitStaticTransposed(std::ifstream &mfile);
		
		/**
		 * Writes a matrix to a csv file
		 * \param filename Name of the csv file
		 */
		void writeCSV(const std::string &filename);
		
		/**
		 * Writes a matrix to a csv file
		 * \param filename Name of the csv file
		 * \param rowheaders Vector with the row headers
		 * \param colheaders Vector with the column headers
		 */
		void writeCSVwHeaders(const std::string &filename, const std::vector<std::string> &rowheaders, const std::vector<std::string> &colheaders);
		
		/**
		 * Writes a matrix to a binary file
		 * \param filename Name of the binary file
		 */
		void writeBinaryByColumns(const std::string &filename);
		
		/** Transposes the matrix, swapping rows and columns.
		 * Whenever the matrix is square, this is done efficiently by avoiding to re-allocate the memory.
		 * For non-square matrices a new matrix is reserved in memory and the old one is deleted.
		 */
		void transpose();

private:
    int nRows; ///< Number of rows
    int nCols; ///< Number of columns
    T ** contents; ///< Matrix contents
};

template<typename T>
Matrix<T>::Matrix() : nRows(0), nCols(0), contents(0) {}

template <typename T>
Matrix<T>::Matrix(int row, int col) : nRows(row), nCols(col), contents(0) {

	if (row == 0 || col == 0)
	{
		nRows = 0;
		nCols = 0;
		contents = 0;
	}
	else
	{
		contents = new T* [nRows];
		contents[0]= new T [nRows*nCols];

		for (int i=1;i<nRows;++i)
			contents[i] = contents[i-1]+nCols;
	}
 }

template <typename T>
Matrix<T>::Matrix(int row, int col,  const T &value) : nRows(row), nCols(col), contents(0) {

	if (row == 0 || col == 0)
	{
		nRows = 0;
		nCols = 0;
		contents = 0;
	}
	else
	{
		contents = new T* [nRows];
		contents[0]= new T [nRows*nCols];

		for (int i=1;i<nRows;i++)
		contents[i] = contents[i-1]+nCols;
	}
	
	fill(value);
 }
 
 

template <typename T>
Matrix<T>::Matrix(T *v, int row, int col) : nRows(row), nCols(col) {

	if (row == 0 || col == 0)
	{
		nRows = 0;
		nCols = 0;
		contents = 0;
	}
	else
	{
		contents = new T* [nRows];
		contents[0]= v;

		for (int i=1;i<nRows;i++)
			contents[i] = contents[i-1]+nCols;
	}
 }

template <typename T>
Matrix<T>::~Matrix(){

	clear();

}

template <typename T>
Matrix<T>::Matrix(const Matrix<T>& o) : nRows(o.nRows), nCols(o.nCols), contents(0) {

	if (o.contents == 0) {
		nRows = 0;
		nCols = 0;
		contents = 0;
		return;
	}

	contents = new T* [nRows];
	contents[0]= new T [nRows*nCols];

	for (int i=1;i<nRows;++i)
		contents[i] = contents[i-1]+nCols;

	for(int i=0; i<nRows*nCols; i++)
		contents[0][i]=o.contents[0][i];
}



template <typename T>
void Matrix<T>::swap(Matrix<T>& o) {
	
	int tmp;
	
	tmp = nRows;
	nRows = o.nRows;
	o.nRows = tmp;
	
	tmp = nCols;
	nCols = o.nCols;
	o.nCols = tmp;
	
	T **tmpp;
	
	tmpp = contents;
	contents = o.contents;
	o.contents = tmpp;
}


template <typename T>
void Matrix<T>::transpose() {
	
	// If the matrix is square, the process is simple
	if(nRows == nCols)
	{
		for(int i=0; i < nRows; i++)
			for(int j = 0; j < i; j++)
				std::swap(contents[i][j], contents[j][i]);
	}
	
	// For a non-square matrix, it's much more complex
	else
	{
		// Create new row pointers
// 		T *newcontents = new T [nCols];
// 		
// 		newcontents[0] = contents[0];
// 		for (int i=1;i<nCols;i++)
// 			newcontents[i] = newcontents[i-1]+nRows;
// 		
// 		// Create binary array to check the cycles
// 		checked = std::bitset<>
// 		
// 		// Transpose data by cycles
// 		while(true)
// 		{
// 			
// 		}
		
		Matrix<T> new_matrix(nCols, nRows);
		
		for(int i=0; i < nRows; i++)
			for(int j = 0; j < nCols; j++)
				new_matrix.contents[j][i] = contents[i][j];
		
		swap(new_matrix);
	}
	
// 	for(int i = 0; i < nRows; ++i)
// 		for(int j = 0; j < nCols; ++j)
// 		{
// 			
// 			T tmp = newcontents[i][j];
// 			newcontents[i][j] = contents[j][i];
// 			contents[j][i] = tmp;
// 		}
	
// 	std::swap(nRows, nCols);
// 	
// 	if(nRows != nCols)
// 		delete [] contents;
// 	
// 	contents = newcontents;
}


template <class T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T> &o) {

	if (this == &o)
		return *this;
	else if (o.contents == 0)
	{
		clear();
		return *this;
	}

	const int MxN = o.nRows * o.nCols;

	if (MxN != nRows * nCols)
	{
		clear();
		contents = new T* [o.nRows];
		contents[0]= new T [MxN];

		for (int i=1; i<o.nRows; i++)
			contents[i] = contents[i-1]+o.nCols;
	}

	nRows = o.nRows;
	nCols = o.nCols;

	for(int i=0; i<MxN; i++)
		contents[0][i]=o.contents[0][i];

	return *this;
}

template <typename T>
void Matrix<T>::clear() {
	if ( contents != 0 ){
		delete [] contents[0];
		delete [] contents;
		contents = 0;
	}

	nRows = 0;
	nCols = 0;
}

template <typename T>
void Matrix<T>::resize(int rows, int columns)
{
	if(rows == 0 || columns == 0)
	{
		clear();
	}
	else if(columns == nCols && rows < nRows)
	{
		nRows = rows;
	}
	else if(columns*rows < size())
	{
		nRows = rows;
		nCols = columns;
		
		for (int i=1; i < nRows; ++i)
			contents[i] = contents[i-1]+nCols;
	}
	else if(rows != nRows || columns != nCols)
	{
		T ** new_matrix;
		
		new_matrix   = new T* [rows]; // rows
		new_matrix[0]= new T  [rows*columns];
		
		for (int i=1; i < rows; ++i)
			new_matrix[i] = new_matrix[i-1]+columns;
		
		if(!empty())
		{
			// copy data from saved pointer to new arrays
			int minrows = std::min<int>(rows, nRows);
			int mincols = std::min<int>(columns, nCols);
			
			for ( int x = 0 ; x < minrows ; x++ )
				memcpy(new_matrix[x], contents[x], mincols * sizeof(T));
			
			clear();
		}
		
		contents = new_matrix;
		
		nRows = rows;
		nCols = columns;
	}
}

template <typename T>
inline T* Matrix<T>::operator[] (int i) {return contents[i];}

template <typename T>
inline T& Matrix<T>::operator() (int i, int j) {return contents[i][j];}

template <typename T>
inline const T* Matrix<T>::operator[] (int i) const {return contents[i];}

template <typename T>
inline const T& Matrix<T>::operator() (int i, int j) const {return contents[i][j];}

template <typename T>
inline int Matrix<T>::rows() const { return nRows; }

template <typename T>
inline int Matrix<T>::cols() const { return nCols; }

template <typename T>
inline int Matrix<T>::size() const { return nRows * nCols; }

template <typename T>
inline bool Matrix<T>::empty() const { return (nRows == 0 || nCols == 0); }

template <typename T>
inline void Matrix<T>::fill(const T &value)
{
	memset(contents[0], value, nRows * nCols * sizeof(T));
}


/**
 * Output operator <<
 * \param out Output stream
 * \param M Matrix to print
 * \return The output stream
 */
template <class T>
std::ostream& operator<<(std::ostream& out, const Matrix<T>& M) {

    out << "Matrix size: (" << M.nRows << "," << M.nCols << ")" << std::endl;
    for(int i=0; i<M.nRows;i++){
        for(int j=0; j<M.nCols;j++){
            out << M.contents[i][j] << " ";
        }
        out << std::endl;
    }

    return out;
}

template <typename T>
inline T** Matrix<T>::getPointer() { return contents; }

template <typename T>
inline T* Matrix<T>::asVector() { return (contents == 0) ? 0 : contents[0]; }


// template <typename T>
// void Matrix<T>::readFromCSV(const std::string &filename, bool header, int numcols)
// {
// 	// Dummy variables to bypass C++ limitations
// 	std::vector<std::string> class_names;
// 	std::vector<int> classes;
// 	
// 	// If the file is gzipped, convert it first
// 	if(filename.find(".gz") == filename.size()-3)
// 	{
// 		std::ifstream file(filename.c_str(), std::ios_base::in | std::ios_base::binary);
// 		
// 		if(!file)
// 		{
// 			std::cerr << "Matrix::readFromCSV: Error when reading file " << filename << std::endl;
// 			return;
// 		}
// 		
// 		boost::iostreams::filtering_streambuf<boost::iostreams::input> inbuf;
// 		inbuf.push(boost::iostreams::gzip_decompressor());
// 		inbuf.push(file);
// 		
// 		//Convert streambuf to istream
// 		std::istream mfile(&inbuf);
// 		
// 		std::string str(static_cast<std::stringstream const&>(std::stringstream() << mfile.rdbuf()).str());
// 		
// 		
// 		
// 		// 		char *fcontents vecArg.assign(std::istreambuf_iterator<char>{&inbuf}, {});;
// 		
// 		
// 		file.close();
// 		
// 		// I abuse the interface of class string to avoid copying it entirely
// 		readFromCSV(&(str[0]), str.size(), class_names, classes, header, numcols);
// 	}
// 	else
// 	{
// 		FILE *f = fopen(filename.c_str(), "rb");
// 		
// 		if(!f)
// 		{
// 			std::cerr << "Matrix::readFromCSV: Error when reading file " << filename << std::endl;
// 			return;
// 		}
// 		
// 		// Read the entire file at once into memory
// 		fseek(f, 0, SEEK_END);
// 		long fsize = ftell(f);
// 		fseek(f, 0, SEEK_SET);
// 		
// 		char *fcontents = (char *)malloc(fsize + 1);
// 		fread(fcontents, fsize, 1, f);
// 		fclose(f);
// 		
// 		fcontents[fsize] = 0;
// 		
// 		readFromCSV(fcontents, fsize, class_names, classes, header, numcols);
// 	}
// }

template <typename T>
void Matrix<T>::readFromCSV(const std::string &filename, bool header, int numcols)
{
	// Dummy variables to bypass C++ limitations
	std::vector<std::string> class_names;
	std::vector<int> classes;
	
	readFromCSV(filename, class_names, classes, header, numcols);
}


template <typename T>
void Matrix<T>::readFromCSV(std::istream &mfile, const std::vector<std::string> &class_names, std::vector<int> &classes, bool header, int numcols, int numlines)
{
	std::string buffer, token;
	
	// This vector will store the classes from the file,
	// which will then be converted to integers
	bool withclass = !class_names.empty();
	
	
	if(numlines < 0)
	{
		numlines = std::count(std::istreambuf_iterator<char>(mfile),
													std::istreambuf_iterator<char>(), '\n');
		
		// Reset the flags and position of the file
		mfile.clear();
		mfile.seekg(0, std::ios::beg);
		
		if(header)
			numlines--;
	}
														
	std::vector<std::string> classes_str(numlines);
		
	if(numcols < 0)
	{
		// Read first line
		getline(mfile, buffer);
		
		numcols = std::count(buffer.begin(), buffer.end(), ',') + 1;
		
		if(withclass)
			numcols--;
		
		mfile.clear();
		mfile.seekg(0, std::ios::beg);
	}
	else if(header)
		getline(mfile, buffer);
	
	// Initialize the matrix
	this->resize(numlines, numcols);

	std::stringstream ss;

	for(int i = 0; i < numlines && mfile; ++i)
	{
		getline(mfile, buffer);
		ss.str(buffer);
		ss.seekg(0, std::ios::beg);

		for(int j = 0; j < numcols; ++j)
		{
			std::getline(ss, token, ',');
			contents[i][j] = atof(token.c_str());
		}
		
		// Whatever is still in the line is the class
		if(withclass)
			std::getline(ss, classes_str[i], ',');
	}
	
	if(withclass)
	{
		classes.resize(classes_str.size());
		
		// Convert the classes to integers	
		if(!isInteger(classes_str[0]))
		{
			for(unsigned int i = 0; i < classes_str.size(); ++i)
			{
				classes_str[i].erase(std::remove(classes_str[i].begin(), classes_str[i].end(), '\"'), classes_str[i].end());
				
				for(unsigned int j = 0; j < class_names.size(); ++j)
					if(class_names[j] == classes_str[i])
					{
						classes[i] = j;
						break;
					}
			}
		}
		else
			// Convert from 1-index to 0-index
			for(unsigned int i = 0; i < classes_str.size(); ++i)
				classes[i] = atoi(classes_str[i].c_str()) - 1;
	}
}



// template <typename T>
// void Matrix<T>::readFromCSV(const std::string &filename, const std::vector<std::string> &class_names, std::vector<int> &classes, bool header, int numcols)
// {
// 	FILE *f = fopen(filename.c_str(), "rb");
// 	
// 	if(!f)
// 	{
// 		std::cerr << "Matrix::readFromCSV: Error when reading file " << filename << std::endl;
// 		return;
// 	}
// 	
// 	// Read the entire file at once into memory
// 	fseek(f, 0, SEEK_END);
// 	long fsize = ftell(f);
// 	fseek(f, 0, SEEK_SET);
// 	
// 	char *fcontents = (char *)malloc(fsize + 1);
// 	fread(fcontents, fsize, 1, f);
// 	fclose(f);
// 	
// 	fcontents[fsize] = 0;
// 	
// 	readFromCSV(fcontents, fsize, class_names, classes, header, numcols);
// }

template <typename T>
void Matrix<T>::readFromCSV(const std::string &filename, const std::vector<std::string> &class_names, std::vector<int> &classes, bool header, int numcols)
{
	
	// If the file is gzipped, convert it first
	if(filename.find(".gz") == filename.size()-3)
	{
		std::ifstream file(filename.c_str(), std::ios_base::in | std::ios_base::binary);
		
		if(!file)
		{
			std::cerr << "Matrix::readFromCSV: Error when reading file " << filename << std::endl;
			return;
		}
		
		// Quick and dirty fix to gzip not being seekable: I go through the file twice
		
		boost::iostreams::filtering_streambuf<boost::iostreams::input> inbuf;
		inbuf.push(boost::iostreams::gzip_decompressor());
		inbuf.push(file);
		
		//Convert streambuf to istream
		std::istream mfile(&inbuf);
		
		int numlines = std::count(std::istreambuf_iterator<char>(mfile),
															std::istreambuf_iterator<char>(), '\n');
		
		if(header)
			numlines--;
		
		file.seekg(0);
		
		boost::iostreams::filtering_streambuf<boost::iostreams::input> inbuf2;
		inbuf2.push(boost::iostreams::gzip_decompressor());
		inbuf2.push(file);
		
		//Convert streambuf to istream
		std::istream mfile2(&inbuf2);
		
		readFromCSV(mfile2, class_names, classes, header, numcols, numlines);
		inbuf2.reset();
		file.close();
	}
	else
	{
		std::ifstream mfile(filename.c_str());
		
		readFromCSV(mfile, class_names, classes, header, numcols);
	}
}

template <typename T>
void Matrix<T>::readFromCSV(char *fcontents, long fsize, const std::vector<std::string> &class_names, std::vector<int> &classes, bool header, int numcols)
{
	int numlines = std::count(fcontents, fcontents+fsize, '\n');
	
	// Cursors to move through the lines
	char *beglinepos  = fcontents;	// Points to the first character of current line
	char *endlinepos = fcontents;	// Points to the last character of current line (\n)
	
	// If the number of columns was not provided we must get it from the first line
	if(numcols == -1 && header)
	{
		endlinepos = strchr(beglinepos, '\n');
		numcols = std::count(beglinepos, endlinepos, ',');
		beglinepos = endlinepos+1;
		--numlines;
	}
	// Skip the header if necessary
	else if(numcols > 0 && header)
	{
		beglinepos = strchr(beglinepos, '\n')+1;
		--numlines;
	}
	else if(numcols == -1 && !header)
	{
		numcols = std::count(beglinepos, strchr(beglinepos, '\n'), ',');
	}
	
	// Initialize the matrix
	this->resize(numlines, numcols);
	
	// This vector will store the classes from the file,
	// which will then be converted to integers
	bool withclass = !class_names.empty();
	std::vector<std::string> classes_str(numlines);
	
	char *begfieldpos = 0;	// Points to first character of the field
	char *endfieldpos = 0;	// Points to separator at the end of the current field
	
	for(int i = 0; i < nRows; ++i)
	{
		// Initialize cursors
		begfieldpos = beglinepos;
		endfieldpos = strchr(begfieldpos, ',');
		endlinepos = strchr(beglinepos, '\n');
		
		for(int j = 0; j < numcols; ++j)
		{
			// I introduce an end of string so that atof reads only one number
			*endfieldpos = 0;
			contents[i][j] = atof(begfieldpos);
			
			// Prepare cursors for next field
			begfieldpos = endfieldpos+1;
			endfieldpos = strchr(begfieldpos, ',');
		}
		
		// Whatever is still in the line is the class
		if(withclass)
		{
			*endlinepos = 0;
			classes_str[i] = std::string(begfieldpos);
		}
		
		// Move to next line
		beglinepos = endlinepos+1;
	}
	
	if(withclass)
	{
		classes.resize(classes_str.size());
		
		// Convert the classes to integers	
		if(!isInteger(classes_str[0]))
		{
			for(unsigned int i = 0; i < classes_str.size(); ++i)
			{
				classes_str[i].erase(std::remove(classes_str[i].begin(), classes_str[i].end(), '\"'), classes_str[i].end());
				
				for(unsigned int j = 0; j < class_names.size(); ++j)
					if(class_names[j] == classes_str[i])
					{
						classes[i] = j;
						break;
					}
			}
		}
		else
			// Convert from 1-index to 0-index
			for(unsigned int i = 0; i < classes_str.size(); ++i)
				classes[i] = atoi(classes_str[i].c_str()) - 1;
	}
}


template <typename T>
void Matrix<T>::readFromCSVRowLimitStatic(std::ifstream &mfile)
{
	std::string buffer, token;
	std::stringstream ss;
	
	int currentrows = 0;
	
	for(int i = 0; i < nRows && mfile; ++i)
	{
		mfile >> buffer;
		
		if(mfile)
		{
			ss.str(buffer);
			ss.seekg(0, std::ios::beg);
			
			for(int j = 0; j < nCols; ++j)
			{
				std::getline(ss, token, ',');
				
// 				if(token == "NA")
// 					contents[i][j] = std::numeric_limits<double>::min();
// 				else
					contents[i][j] = atof(token.c_str());
			}
			
			currentrows++;
		}
	}
	
	if(currentrows < nRows)
		nRows = currentrows;
}



template <typename T>
void Matrix<T>::readFromCSVRowLimitStaticTransposed(std::ifstream &mfile)
{
	std::string buffer, token;
	std::stringstream ss;
	
	int currentrows = 0;
	
	for(int i = 0; i < nCols && mfile; ++i)
	{
		mfile >> buffer;
		
		if(mfile)
		{
			ss.str(buffer);
			ss.seekg(0, std::ios::beg);
			
			for(int j = 0; j < nRows; ++j)
			{
				std::getline(ss, token, ',');
				
				// 				if(token == "NA")
				// 					contents[i][j] = std::numeric_limits<double>::min();
				// 				else
				contents[j][i] = atof(token.c_str());
			}
			
			currentrows++;
		}
	}
	
	if(currentrows < nCols)
		nCols = currentrows;
}

template <typename T>
void Matrix<T>::writeCSV(const std::string &filename)
{
	std::ofstream mfile(filename.c_str());
	
	if(!mfile)
	{
		std::cerr << "Matrix::writeCSV: Error when opening file " << filename  << " for output " << std::endl;
		return;
	}
	
	for(int i = 0; i < nRows; ++i)
	{
		mfile << contents[i][0];
		
		for(int j = 1; j < nCols; ++j)
			mfile << "," << contents[i][j];
		
		mfile << std::endl;
	}
	
	mfile.close();
}



template <typename T>
void Matrix<T>::writeBinaryByColumns(const std::string &filename)
{
	std::ofstream mfile(filename.c_str(), std::ofstream::binary);
	
	if(!mfile)
	{
		std::cerr << "Matrix::writeBinaryByColumns: Error when opening file " << filename  << " for output " << std::endl;
		return;
	}
	
	// Write the size
	mfile.write(reinterpret_cast<char *>( &nRows ), sizeof(nRows));
	mfile.write(reinterpret_cast<char *>( &nCols ), sizeof(nCols));
	
	for(int j = 0; j < nCols; ++j)
		for(int i = 0; i < nRows; ++i)
			mfile.write(reinterpret_cast<char *> (&contents[i][j]), sizeof(T));
	
	mfile.close();
}


template <typename T>
void Matrix<T>::writeCSVwHeaders(const std::string &filename, const std::vector<std::string> &rowheaders, const std::vector<std::string> &colheaders)
{
	std::ofstream mfile(filename.c_str());
	
	if(!mfile)
	{
		std::cerr << "Matrix::writeCSVwHeaders: Error when opening file " << filename  << " for output " << std::endl;
		return;
	}
	
	if(rowheaders.size() != (unsigned int) nRows || colheaders.size() != (unsigned int) nCols)
	{
		std::cerr << "Matrix::writeCSVwHeaders: The headers have the wrong size: " << rowheaders.size() << "x" << colheaders.size() << " for a matrix of size " << nRows << "x " << nCols << std::endl;
		return;
	}
	
	// Write the header row
	for(unsigned int i = 0; i < colheaders.size(); ++i)
		mfile << "," << colheaders[i];
	mfile << std::endl;
	
	for(int i = 0; i < nRows; ++i)
	{
		mfile << rowheaders[i];
		
		for(int j = 0; j < nCols; ++j)
			mfile << "," << contents[i][j];
		
		mfile << std::endl;
	}
	
	mfile.close();
}



#endif
