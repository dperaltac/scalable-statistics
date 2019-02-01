/**
 * \file    Hyperplane2D.h
 * \author  Daniel Peralta <daniel.peralta@irc.vib-ugent.be>
 * \version 1.0
 *
 * \section DESCRIPTION_HYPERPLANE2D
 *
 * Class that implements an efficient reference to a two dimensional hyperplane within a \p Tensor
 */

#ifndef HYPERPLANE2D_H
#define HYPERPLANE2D_H

/**
 * @class Hyperplane2D
 *
 * The Hyperplane2D class allow us to extract an 2 dimensional slice of an higher dimensional Tensor.
 * This is done by reference, allowing the original Tensor to be efficiently accessed and modified in a lower dimensionality space.
 * A Hyperplane is itself a Tensor and can in turn be sliced into further hyperplanes.
 * The advantage of \p Hyperplane2D w.r.t. \p Hyperplane is that it offers efficient access with two indices.
 */
template<class T>
class Hyperplane2D : public Hyperplane<T> {
	
public:
	
	/** Default destructor */
	virtual ~Hyperplane2D() {
		// This stops the superclass destructor from deleting the whole contents
		this->contents = 0;
	}
	
	/**
	 * This method is overriden and does nothing, because the Hyperplane is a reference to a slice of a Tensor that shouldn't be modified in this way
	 */
	void clear() {};
	
	/**
	 * Copy constructor
	 * \param H Object to copy from
	 */
	Hyperplane2D(const Hyperplane2D<T> &H);
	
	/**
	 * Constructor
	 * \param m Tensor to create the hyperplane from
	 * \param indices Indices of the target hyperplane
	 */
	Hyperplane2D(const Tensor<T> &m, std::initializer_list<int> indices);
	
	/**
	 * Constructor
	 * \param m Tensor to create the hyperplane from
	 * \param indices Indices of the target hyperplane
	 */
	Hyperplane2D(Tensor<T> &m, std::initializer_list<int> indices);
	
	
	/**
	 * Assignment operator
	 * \param H Object to assign from
	 * \return A reference to this object
	 */
	Hyperplane2D<T> &operator=(const Hyperplane2D<T> &H);
	
	/**
	 * Efficient access operator
	 * \param i Row
	 * \param j Column
	 * \return Returns element (\p i, \p j) of the hyperplane
	 */
	T& operator() (int i, int j);
	
	/**
	 * Efficient access operator
	 * \param i Row
	 * \param j Column
	 * \return Returns element (\p i, \p j) of the hyperplane
	 */
	const T& operator() (int i, int j) const;
	
protected:
	
	Hyperplane2D() : Hyperplane<T>() {};
	
	int nRows;	///< Number of rows
	int nCols;	///< Number of columns
};


template <typename T>
inline T& Hyperplane2D<T>::operator() (int i, int j) {return this->contents[i*this->nCols + j];}


template <typename T>
inline const T& Hyperplane2D<T>::operator() (int i, int j) const {return this->contents[i*this->nCols + j];}




template <typename T>
Hyperplane2D<T>::Hyperplane2D(const Hyperplane2D<T> &H)
{
	this->dims       = H.dims;
	this->dimsize    = H.dimsize;
	this->contents   = H.contents;
	this->total_size = H.total_size;
	this->nRows      = H.nRows;
	this->nCols      = H.nCols;
}

template <typename T>
Hyperplane2D<T> & Hyperplane2D<T>::operator=(const Hyperplane2D<T> &H)
{
	if(this != &H)
	{
		this->dims       = H.dims;
		this->dimsize    = H.dimsize;
		this->contents   = H.contents;
		this->total_size = H.total_size;
		this->nRows      = H.nRows;
		this->nCols      = H.nCols;
	}
	
	return *this;
}


template <typename T>
Hyperplane2D<T>::Hyperplane2D(const Tensor<T> &m, std::initializer_list<int> indices)
{
	unsigned int c = 0, pos = 0;
	
	for(auto i : indices)
	{
		pos += i*m.dimsize[c];
		c++;
		
		assert(c <= m.dimsize.size());
	}
	
	assert(m.dims.size()-c == 2);
	
	this->dims    = std::vector<int>(m.dims.size()-c);
	this->dimsize = std::vector<int>(m.dims.size()-c);
	
	for(unsigned int j = c; j < m.dims.size(); ++j)
	{
		this->dims   [j-c] = m.dims[j];
		this->dimsize[j-c] = m.dimsize[j];
	}
	
	this->nRows      = this->dims[0];
	this->nCols      = this->dims[1];
	
	this->contents = m.contents + pos;
	this->total_size = m.dimsize[c-1];
}


template <typename T>
Hyperplane2D<T>::Hyperplane2D(Tensor<T> &m, std::initializer_list<int> indices)
{
	unsigned int c = 0, pos = 0;
	
	for(auto i : indices)
	{
		pos += i*m.dimsize[c];
		c++;
		
		assert(c <= m.dimsize.size());
	}
	
	assert(m.dims.size()-c == 2);
	
	this->dims    = std::vector<int>(m.dims.size()-c);
	this->dimsize = std::vector<int>(m.dims.size()-c);
	
	for(unsigned int j = c; j < m.dims.size(); ++j)
	{
		this->dims   [j-c] = m.dims[j];
		this->dimsize[j-c] = m.dimsize[j];
	}
	
	this->nRows      = this->dims[0];
	this->nCols      = this->dims[1];
	
	this->contents = m.contents + pos;
	this->total_size = m.dimsize[c-1];
}

#endif
