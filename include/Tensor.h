/**
 * \file    Tensor.h
 * \author  Daniel Peralta <daniel.peralta@irc.vib-ugent.be>
 * \version 1.0
 *
 * \section DESCRIPTION_MULTIARRAY
 *
 * A template for building generic multidimensional arrays
 */

#ifndef MULTIARRAY_H
#define MULTIARRAY_H

#include<iostream>
#include <vector>
#include <unistd.h>
#include <cassert>
#include <algorithm>

#include "Matrix.h"
// #include "Functions.h"

template<typename T> class Tensor;
template<typename T> class Hyperplane;
template<typename T> class Hyperplane2D;
template<typename T> std::ostream& operator<<(std::ostream& out, const Tensor<T>& M);

template <typename T>

/**
 * @class Tensor
 *
 * The Tensor class allows to define generic matrixes, able to store any of the primitive types.
 * It supports also some additional functions, such as filling or printing (<<).
 */
class Tensor{
	
	friend Hyperplane<T>;
	friend Hyperplane2D<T>;
	
public:
	
	// iterator
	typedef T* iterator;	///< Iterator
	typedef const T* const_iterator;	///< Constant iterator
	
	// iterator functions
	/**
	 * Iterator pointing to the first element of the tensor
	 * \return Iterator pointing to the first element of the tensor
	 */
	inline iterator begin() { return contents; }
	
	/**
	 * Iterator pointing to the end of the tensor
	 * \return Iterator pointing to end of the tensor
	 */
	inline iterator end()   { return contents + total_size; }
	
	/**
	 * Iterator pointing to the first element of the tensor
	 * \return Iterator pointing to the first element of the tensor
	 */
	inline const_iterator begin() const { return contents; }
	
	/**
	 * Iterator pointing to the end of the tensor
	 * \return Iterator pointing to end of the tensor
	 */
	inline const_iterator end() const   { return contents + total_size; }
	
	/**
	 * Iterator pointing to the beginining of hyperplane \p i, measured in the first dimension of the tensor
	 * \param i Hyperplane pointed at
	 * \return Iterator pointing to hyperplane \p i
	 */
	inline iterator begin(int i) { return contents + i*dimsize[0]; }
	
	/**
	 * Iterator pointing to the end of hyperplane \p i, measured in the first dimension of the tensor
	 * \param i Hyperplane pointed at
	 * \return Iterator pointing to hyperplane \p i
	 */
	inline iterator end(int i)   { return contents + (i+1)*dimsize[0]; }
	
	/**
	 * Iterator pointing to the beginining of hyperplane \p i, measured in the first dimension of the tensor
	 * \param i Hyperplane pointed at
	 * \return Iterator pointing to hyperplane \p i
	 */
	inline const_iterator begin(int i) const { return contents + i*dimsize[0]; }
	
	/**
	 * Iterator pointing to the end of hyperplane \p i, measured in the first dimension of the tensor
	 * \param i Hyperplane pointed at
	 * \return Iterator pointing to hyperplane \p i
	 */
	inline const_iterator end(int i) const   { return contents + (i+1)*dimsize[0]; }
	

    /** Default Constructor */
		Tensor();
		
		/**
		 * Constructor
		 * \param dimensions List of sizes of each dimension
		 */
		Tensor(std::initializer_list<int> dimensions);
		
		/**
		 * Constructor
		 * \param dimensions List of sizes of each dimension
		 */
		Tensor(const std::vector<int> &dimensions);

	/** Default destructor */
	virtual ~Tensor();

	/**
		* Copy constructor
		* \param o Object to copy from
		*/
	Tensor(const Tensor& o);

	/**
		* Assignment operator
		* \param o Object to assign from
		* \return A reference to this object
		*/
	Tensor & operator= (const Tensor &o);

	/**
		* Erase all the contents of the array
		*/
	void clear();

	/**
		* Access operator
		* \param indices Contains the index accessed for every dimension
		* \return A reference the value stored in (indices[0], ... indices[dimnum()-1])
		*/
	const T& operator() (std::initializer_list<int> indices) const;
	
	/**
	 * Access operator
	 * \param indices Contains the index accessed for every dimension
	 * \return A reference the value stored in (indices[0], ... indices[dimnum()-1])
	 */
	T& operator() (std::initializer_list<int> indices);
	
	/**
	 * Hyperplane. Returns a hyperplane of the current Tensor with 1 dimension less.
	 * \param i Index
	 * \return A Tensor corresponding to a hyperplane of the current Tensor with 1 dimension less.
	 */
	Hyperplane<T> hyperplane (int i);
	
	/**
	 * Hyperplane. Returns a hyperplane of the current Tensor with 1 dimension less.
	 * \param i Index
	 * \return A Tensor corresponding to a hyperplane of the current Tensor with 1 dimension less.
	 */
	Hyperplane<T> hyperplane (int i) const;
	
	/**
	 * Returns a lower dimensionality slice of the current Tensor, determined by \p indices, which correspond to the higher dimensions of the Tensor.
	 * \param indices List of indices that determine the slice.
	 * \return A Tensor corresponding to a hyperplane of the current Tensor with indices.size() dimensions less.
	 */
	Hyperplane<T> hyperplane (std::initializer_list<int> indices);
	
	/**
	 * Returns a lower dimensionality slice of the current Tensor, determined by \p indices, which correspond to the higher dimensions of the Tensor.
	 * \param indices List of indices that determine the slice.
	 * \return A Tensor corresponding to a hyperplane of the current Tensor with indices.size() dimensions less.
	 */
	Hyperplane<T> hyperplane (std::initializer_list<int> indices) const;
	
	/**
	 * Hyperplane. Returns a hyperplane of the current Tensor with 1 dimension less.
	 * \param i Index
	 * \return A Tensor corresponding to a hyperplane of the current Tensor with 1 dimension less.
	 */
	Hyperplane2D<T> hyperplane2D (int i);
	
	/**
	 * Hyperplane. Returns a hyperplane of the current Tensor with 1 dimension less.
	 * \param i Index
	 * \return A Tensor corresponding to a hyperplane of the current Tensor with 1 dimension less.
	 */
	Hyperplane2D<T> hyperplane2D (int i) const;
	
	/**
	 * Returns a lower dimensionality slice of the current Tensor, determined by \p indices, which correspond to the higher dimensions of the Tensor.
	 * \param indices List of indices that determine the slice.
	 * \return A Tensor corresponding to a hyperplane of the current Tensor with indices.size() dimensions less.
	 */
	Hyperplane2D<T> hyperplane2D (std::initializer_list<int> indices);
	
	/**
	 * Returns a lower dimensionality slice of the current Tensor, determined by \p indices, which correspond to the higher dimensions of the Tensor.
	 * \param indices List of indices that determine the slice.
	 * \return A Tensor corresponding to a hyperplane of the current Tensor with indices.size() dimensions less.
	 */
	Hyperplane2D<T> hyperplane2D (std::initializer_list<int> indices) const;
	
	/**
	 * Hyperplane. Returns a newly reserved Tensor that contains a hyperplane of the current Tensor with 1 dimension less.
	 * \param i Index
	 * \param result (output) A newly reserved Tensor corresponding to hyperplane of the current Tensor with 1 dimension less.
	 */
	void newFromHyperplane (Tensor<T> &result, int i) const;
	
	/**
	 * Hyperplane for the 3D case. Returns a newly reserved Matrix that contains a hyperplane of the current Tensor with 1 dimension less.
	 * \param i Index
	 * \param result (output) A newly reserved Tensor corresponding to hyperplane of the current Tensor with 1 dimension less.
	 */
	void newFromHyperplane (Matrix<T> &result, int i) const;
	
	/**
	 * Get the size of dimension \p i of the array
	 * \param i Dimension whose size is queried
	 * \return Size of dimension \p i of the array
	 */
	int dim(int i) const;
	
	/**
	 * Get the number of dimensions
	 * \return Number of dimensions
	 */
	int dimnum() const;
	
	/**
	 * Get the vector of dimensions
	 * \return Vector of dimensions
	 */
	std::vector<int> getDims() const;

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
		 * Fills the matrix with a given value
		 * \param value Value used for filling the matrix
		 */
		void fill(const T &value);
		
		/**
		 * Fills the lower triangular matrix with a given value
		 * \param value Value used for filling the matrix
		 * \todo When \p Matrix is a specialization of Tensor, this method should be moved there. And some method should be created to extract a 2DHyperplane
		 * \todo Use memset
		 */
		void fillLower(const T &value);
		
		/**
		 * Fills the upper triangular matrix with a given value
		 * \param value Value used for filling the matrix
		 * \todo When \p Matrix is a specialization of Tensor, this method should be moved there. And some method should be created to extract a 2DHyperplane
		 * \todo Use memset
		 */
		void fillUpper(const T &value);
		
		/**
		 * Fills the upper triangular matrix and the diagonal with a given value
		 * \param value Value used for filling the matrix
		 * \todo When \p Matrix is a specialization of Tensor, this method should be moved there. And some method should be created to extract a 2DHyperplane
		 * \todo Use memset
		 */
		void fillUpperAndDiag(const T &value);
		
		/**
		 * Reserves memory for the specified dimension sizes. Any contents are destroyed.
		 * \param dimensions Contains the size of every dimension
		 */
		void reserve(std::initializer_list<int> dimensions);
		
		/**
		 * Reserves memory for the specified dimension sizes. Any contents are destroyed.
		 * \param dimensions Contains the size of every dimension
		 */
		void reserve(const std::vector<int> &dimensions);
		
		/**
		 * Outputs the contents of the hyperplane to \p out, starting from position \p pos of dimension \p curdim
		 * \param out Output stream
		 * \param pos Starting position from which the contents are printed
		 * \param curdim Starting dimension. For instance, if (\p curdim == dimnum()-1) a vector is printed.
		 * \return Reference to the stream
		 */
		std::ostream & printHyperplane(std::ostream& out, int pos, int curdim) const;
		

		//friend operators
		friend std::ostream& operator<< <>(std::ostream& out, const Tensor<T>& M);
		
		
		/**
		 * Return the pointer of the matrix, from where all the data can be read as a vector.
		 * \return Pointer to the start of the array
		 */
		T* asVector();
		
		/**
		 * Return the pointer of the matrix, from where all the data can be read as a vector.
		 * \return Constant pointer to the start of the array
		 */
		const T* asVector() const;
		
		
		/**
		 * Efficiently swaps the contents of the array by that of \p o
		 * \param o Tensor to be swapped by \p this
		 */
		void swap(Tensor &o);
		
		
		/**
		 * Fills the array with the values contained in \p M, in the same order, until the minimum size of the two arrays is reached.
		 * \param M Tensor to copy the values from
		 */
		void fillValues(const Tensor<T> &M);

protected:
	
	unsigned int total_size; ///< Total number of elements
	std::vector<int> dims; ///< Size of each dimension
	std::vector<int> dimsize; ///< Number of single elements contained in each dimension and all those under it
	
	T * contents; ///< Tensor contents
};

template<typename T>
Tensor<T>::Tensor() : total_size(0), dims(), dimsize(), contents(0) {}

template <typename T>
Tensor<T>::Tensor(std::initializer_list<int> dimensions) :  contents(0) {
	reserve(dimensions);
}

template <typename T>
Tensor<T>::Tensor(const std::vector<int> &dimensions) : contents(0) {
	reserve(dimensions);
}


template <typename T>
Tensor<T>::~Tensor(){
	clear();
}


template <typename T>
Tensor<T>::Tensor(const Tensor<T>& o) : total_size(o.total_size), dims(o.dims), dimsize(o.dimsize) {

	contents = new T[total_size];
	
	for(unsigned int i = 0; i < total_size; ++i)
		contents[i] = o.contents[i];
}



template <typename T>
void Tensor<T>::swap(Tensor<T>& o) {
	
	dims.swap(o.dims);
	dimsize.swap(o.dimsize);
	std::swap(contents, o.contents);
	std::swap(total_size, o.total_size);
}

template <typename T>
void Tensor<T>::fillValues(const Tensor<T> &M)
{
	unsigned int minsize = std::min(total_size, M.total_size);
	memcpy(contents, M.contents, minsize * sizeof(T));
}



template <class T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T> &o) {

	if (this == &o)
		return *this;
	else if (o.contents == 0)
	{
		clear();
		return *this;
	}
	
	if(total_size != o.total_size)
	{
		clear();
		total_size = o.total_size;
		
		contents = new T[total_size];
	}
	
	dims = o.dims;
	dimsize = o.dimsize;
	
	for(unsigned int i = 0; i < total_size; ++i)
		contents[i] = o.contents[i];

	return *this;
}



template <typename T>
void Tensor<T>::reserve(std::initializer_list<int> dimensions)
{
// 	if(dims == dimensions)
// 		return;
	
	clear();
	
	total_size = 1;
	
	for(auto d : dimensions)
	{
		dims.push_back(d);
		total_size *= d;
	}
	
	dimsize.resize(dims.size());
	dimsize[dimsize.size()-1] = 1;
	
	for(int i = dims.size()-2; i >= 0; --i)
		dimsize[i] = dimsize[i+1] * dims[i+1];
	
	contents = new T[total_size];
}


template <typename T>
void Tensor<T>::reserve(const std::vector<int> &dimensions)
{
// 	if(dims == dimensions)
// 		return;
	
	clear();
	
	total_size = 1;
	
	for(auto d : dimensions)
	{
		dims.push_back(d);
		total_size *= d;
	}
	
	dimsize.resize(dims.size());
	dimsize[dimsize.size()-1] = 1;
	
	for(int i = dims.size()-2; i >= 0; --i)
		dimsize[i] = dimsize[i+1] * dims[i+1];
	
	contents = new T[total_size];
}

template <typename T>
void Tensor<T>::clear()
{
	if ( contents != 0 )
	{
		delete [] contents;
		contents = 0;
	}

	dims.clear();
	dimsize.clear();
	total_size = 0;
}


template <typename T>
T& Tensor<T>::operator() (std::initializer_list<int> indices)
{
	unsigned int c = 0, pos = 0;
	
	for(auto i : indices)
	{
		pos += i*dimsize[c];
		c++;
		
		assert(c <= dimsize.size());
	}
	
	assert(c == dimsize.size());
	
	return contents[pos];
}

template <typename T>
const T& Tensor<T>::operator()  (std::initializer_list<int> indices) const
{
	unsigned int c = 0, pos = 0;
	
	for(auto i : indices)
	{
		pos += i*dimsize[c];
		c++;
		
		assert(c <= dimsize.size());
	}
	
	assert(c == dimsize.size());
	
	return contents[pos];
}


template <typename T>
inline Hyperplane<T> Tensor<T>::hyperplane (int i)
{
	return hyperplane({i});
}


template <typename T>
inline Hyperplane<T> Tensor<T>::hyperplane (int i) const
{
	return hyperplane({i});
}

template <typename T>
inline Hyperplane<T> Tensor<T>::hyperplane (std::initializer_list<int> indices)
{
	return Hyperplane<T>(*this, indices);
}


template <typename T>
inline Hyperplane<T> Tensor<T>::hyperplane (std::initializer_list<int> indices) const
{
	return Hyperplane<T>(*this, indices);
}



template <typename T>
inline Hyperplane2D<T> Tensor<T>::hyperplane2D (int i)
{
	return hyperplane2D({i});
}


template <typename T>
inline Hyperplane2D<T> Tensor<T>::hyperplane2D (int i) const
{
	return hyperplane2D({i});
}

template <typename T>
inline Hyperplane2D<T> Tensor<T>::hyperplane2D (std::initializer_list<int> indices)
{
	return Hyperplane2D<T>(*this, indices);
}


template <typename T>
inline Hyperplane2D<T> Tensor<T>::hyperplane2D (std::initializer_list<int> indices) const
{
	return Hyperplane2D<T>(*this, indices);
}

template <typename T>
void Tensor<T>::newFromHyperplane (Tensor<T> &result, int i) const
{
	std::vector<int> newdims = dims;
	newdims.erase(newdims.begin());
	
	result.reserve(newdims);
	
	for(unsigned int j = 0; j < result.total_size; ++j)
		result.contents[j] = contents[i*dimsize[0] + j];
}

template <typename T>
void Tensor<T>::newFromHyperplane (Matrix<T> &result, int i) const
{
	assert(dims.size() == 3);
	
	result.resize(dims[1], dims[2]);
		
	for(int j = 0; j < dims[1] * dims[2]; ++j)
		result.asVector()[j] = contents[i*dimsize[0] + j];
}

template <typename T>
inline int Tensor<T>::dim(int i) const { return dims[i]; }

template <typename T>
inline int Tensor<T>::dimnum() const { return dims.size(); }

template <typename T>
inline std::vector<int> Tensor<T>::getDims() const { return dims; }

template <typename T>
inline int Tensor<T>::size() const { return total_size; }

template <typename T>
inline bool Tensor<T>::empty() const { return contents == 0; }

template <typename T>
inline void Tensor<T>::fill(const T &value)
{
	memset(contents, value, total_size * sizeof(T));
}

template <typename T>
void Tensor<T>::fillLower(const T &value)
{
	assert(dimsize.size() == 2);
	
	for(int i = 1; i < dims[0]; ++i)
		memset(contents+(i*dimsize[0]), value, i * sizeof(T));
}

template <typename T>
void Tensor<T>::fillUpper(const T &value)
{
	assert(dimsize.size() == 2);
	
	for(int i = 0; i < dims[0]-1; ++i)
		memset(contents+(i*dimsize[0]+i+1), value, (dims[1]-i-1)* sizeof(T));
}

template <typename T>
void Tensor<T>::fillUpperAndDiag(const T &value)
{
	assert(dimsize.size() == 2);
	
	for(int i = 0; i < dims[0]; ++i)
		memset(contents+(i*dimsize[0]+i), value, (dims[1]-i)* sizeof(T));
}


template <class T>
std::ostream & Tensor<T>::printHyperplane(std::ostream& out, int pos, int curdim) const
{
	if(dims.size() - curdim == 1)
	{
		for(int j=0; j<dims[curdim];j++)
			out << contents[pos+j] << " ";
	}
	else if(dims.size() - curdim == 2)
	{
		int rows = dims[curdim];
		int cols = dims[curdim+1];
		
		for(int i=0; i<rows;i++)
		{
			for(int j=0; j<cols;j++)
				out << contents[pos+i*cols+j] << " ";
			out << std::endl;
		}
	}
	else
	{
		for(int i = 0; i < dims[curdim]; ++i)
		{
			out << "Dimension " << dims.size()-curdim << " (elem " << i+1 << " of " << dims[curdim] << "):" << std::endl;
			
			printHyperplane(out, pos + dimsize[curdim]*i, curdim+1);
			out << std::endl;
		}
	}
	
	return out;
}


/**
 * Output operator <<
 * \param out Output stream
 * \param M Tensor to print
 * \return The output stream
 */
template <class T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& M) {

	out << "Tensor size: (";	
	
	for(auto i : M.dims)
		out << i << ",";
	
	out << ")" << std::endl;
	
	return M.printHyperplane(out, 0, 0);
}

template <typename T>
inline T* Tensor<T>::asVector() { return contents; }

template <typename T>
inline const T* Tensor<T>::asVector() const { return contents; }




#endif
