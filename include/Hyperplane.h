/**
 * \file    Hyperplane.h
 * \author  Daniel Peralta <daniel.peralta@irc.vib-ugent.be>
 * \version 1.0
 *
 * \section DESCRIPTION_HYPERPLANE
 *
 * Class that implements an efficient reference to a hyperplane within a \p Tensor
 */

#ifndef HYPERPLANE_H
#define HYPERPLANE_H

#include <vector>
#include <cassert>
#include <algorithm>

#include "Tensor.h"


/**
 * @class Hyperplane
 *
 * The Hyperplane class allow us to extract an n-1 dimensional slice of a higher dimenional Tensor.
 * This is done by reference, allowing the original Tensor to be efficiently accessed and modified in a lower dimensionality space.
 * A Hyperplane is itself a Tensor and can in turn be sliced into further hyperplanes.
 */
template<class T>
class Hyperplane : public Tensor<T> {
	
	friend Tensor<T>;
	
public:
	
	/** Default destructor */
	virtual ~Hyperplane() {
		// This stops the superclass destructor from deleting the whole contents
		this->contents = 0;
	}
	
	/**
	 * Copy constructor
	 * \param H Object to copy from
	 */
	Hyperplane(const Hyperplane<T> &H);
	
	/**
	 * Constructor
	 * \param m Tensor to create the hyperplane from
	 * \param indices Indices of the target hyperplane
	 */
	Hyperplane(const Tensor<T> &m, std::initializer_list<int> indices);
	
	/**
	 * Constructor
	 * \param m Tensor to create the hyperplane from
	 * \param indices Indices of the target hyperplane
	 */
	Hyperplane(Tensor<T> &m, std::initializer_list<int> indices);
	
	
	/**
	 * Assignment operator
	 * \param H Object to assign from
	 * \return A reference to this object
	 */
	Hyperplane<T> &operator=(const Hyperplane<T> &H);
	
	/**
	 * This method is overriden and does nothing, because the Hyperplane is a reference to a slice of a Tensor that shouldn't be modified in this way
	 */
	void clear() {};
	
	/**
	 * This method is overriden and does nothing, because the Hyperplane is a reference to a slice of a Tensor that shouldn't be modified in this way
	 * \param dimensions
	 */
	void reserve(std::initializer_list<int> dimensions) {};
	
	/**
	 * This method is overriden and does nothing, because the Hyperplane is a reference to a slice of a Tensor that shouldn't be modified in this way
	 * \param dimensions
	 */
	void reserve(const std::vector<int> &dimensions) {};
	
protected:
	
	Hyperplane() : Tensor<T>() {};	
};


template <typename T>
Hyperplane<T>::Hyperplane(const Hyperplane<T> &H)
{
	this->dims       = H.dims;
	this->dimsize    = H.dimsize;
	this->contents   = H.contents;
	this->total_size = H.total_size;
}

template <typename T>
Hyperplane<T> & Hyperplane<T>::operator=(const Hyperplane<T> &H)
{
	if(this != &H)
	{
		this->dims       = H.dims;
		this->dimsize    = H.dimsize;
		this->contents   = H.contents;
		this->total_size = H.total_size;
	}
	
	return *this;
}




template <typename T>
Hyperplane<T>::Hyperplane(const Tensor<T> &m, std::initializer_list<int> indices)
{
	unsigned int c = 0, pos = 0;
	
	for(auto i : indices)
	{
		pos += i*m.dimsize[c];
		c++;
		
		assert(c <= m.dimsize.size());
	}
	
	this->dims    = std::vector<int>(m.dims.size()-c);
	this->dimsize = std::vector<int>(m.dims.size()-c);
	
	for(unsigned int j = c; j < m.dims.size(); ++j)
	{
		this->dims   [j-c] = m.dims[j];
		this->dimsize[j-c] = m.dimsize[j];
	}
	
	this->contents = m.contents + pos;
	this->total_size = m.dimsize[c-1];
}


template <typename T>
Hyperplane<T>::Hyperplane(Tensor<T> &m, std::initializer_list<int> indices)
{
	unsigned int c = 0, pos = 0;
	
	for(auto i : indices)
	{
		pos += i*m.dimsize[c];
		c++;
		
		assert(c <= m.dimsize.size());
	}
	
	this->dims    = std::vector<int>(m.dims.size()-c);
	this->dimsize = std::vector<int>(m.dims.size()-c);
	
	for(unsigned int j = c; j < m.dims.size(); ++j)
	{
		this->dims   [j-c] = m.dims[j];
		this->dimsize[j-c] = m.dimsize[j];
	}
	
	this->contents = m.contents + pos;
	this->total_size = m.dimsize[c-1];
}


#endif
