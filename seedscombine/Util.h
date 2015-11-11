// This file contains utility functions for general purpose use

#ifndef SUPERPIXEL_UTIL_H
#define	SUPERPIXEL_UTIL_H

#include <vector>
#include <unordered_map>

#include "Coord.h"

using namespace std;

void sample_mean(vector<float> &values, float *meanPtr);
void sample_mean_delta_squared_div(vector<float> &values, float mean, float *stddevPtr);
vector<float> float_diffs(vector<float> &values);

static inline
int32_t mini(int32_t v1, int32_t v2) {
  if (v1 < v2) {
    return v1;
  } else {
    return v2;
  }
}

static inline
int32_t maxi(int32_t v1, int32_t v2) {
  if (v1 > v2) {
    return v1;
  } else {
    return v2;
  }
}

// http://www.brucelindbloom.com/index.html?Equations.html
// Delta E (CIE 1976)

static inline
double
delta_e_1976(uint8_t L1, uint8_t A1, uint8_t B1,
             uint8_t L2, uint8_t A2, uint8_t B2)
{
  double dL = (double)L1 - (double)L2;
  double dA = (double)A1 - (double)A2;
  double dB = (double)B1 - (double)B2;
  
  dL = dL * dL;
  dA = dA * dA;
  dB = dB * dB;
  
  double delta_E = sqrt(dL + dA + dB);
  return delta_E;
}

// This template util function will copy all the values in a range identified
// by a specific start and end iterator.

template <typename T>
static inline
void
append_to_vector(vector<T> &dst, typename vector<T>::iterator iterBegin, typename vector<T>::iterator iterEnd)
{
  for (typename vector<T>::iterator it = iterBegin; it != iterEnd; ++it ) {
    T val = *it;
    dst.push_back(val);
  }
}

// This template util function will copy all the elements from src into dst.
// The two vectors must contain the same type value.

template <typename T>
static inline
void
append_to_vector(vector<T> &dst, vector<T> &src)
{
  append_to_vector(dst, src.begin(), src.end());
}

// Util method to return the 8 neighbors of a center point in the order
// R, U, L, D, UR, UL, DL, DR while taking the image bounds into
// account. For example, the point (0, 1) will not return UL, L, or DL.
// The iteration order here is implicitly returning all the pixels
// that are sqrt(dx,dy)=1 first and then the 1.4 distance pixels last.

const vector<Coord>
get8Neighbors(Coord center, int32_t width, int32_t height);

// Given a binary image consisting of zero or non-zero values, calculate
// the center of mass for a shape defined by the white pixels.

void centerOfMass(uint8_t *bytePtr, uint32_t width, uint32_t height, uint32_t *xPtr, uint32_t *yPtr);

// Given a vector of coordinate values and weights associated with the coordinates
// sort the coordinate values in terms of the integer weights in descending order.

typedef tuple<Coord, int32_t> CoordIntWeightTuple;

void
sortCoordIntWeightTuples(vector<CoordIntWeightTuple> &tuples, bool decreasing);

// adler checksum

uint32_t my_adler32(
                    uint32_t adler,
                    unsigned char const *buf,
                    uint32_t len,
                    uint32_t singleCallMode);

// Treat an unsigned byte as a signed integer and then square
// the value so that it becomes an unsigned integer. This is
// the same as (abs(v) + abs(v)).

static inline
uint32_t squareAsSignedByte(uint8_t bval) {
  int32_t val = (int8_t)bval;
  val = val * val;
  return (uint32_t) val;
}

#endif // SUPERPIXEL_UTIL_H
