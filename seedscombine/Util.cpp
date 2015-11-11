// This file contains utility functions for general purpose use and interaction with OpenCV

#include "Util.h"

#include <ostream>
#include <iostream>

using namespace std;

// mean of N values

void sample_mean(vector<float> &values, float *meanPtr) {
  int len = (int) values.size();
  if (len == 0) {
    *meanPtr = 0.0f;
    return;
  } else if (len == 1) {
    *meanPtr = values[0];
    return;
  }
  
  float sum = 0.0f;
  for (vector<float>::iterator it = values.begin(); it != values.end(); ++it) {
    float val = *it;
    sum += val;
  }
  if (sum == 0.0f) {
    *meanPtr = 0.0f;
  } else {
    *meanPtr = sum / len;
  }
}

// The caller must pass in the mean value calculated via sample_mean()

void sample_mean_delta_squared_div(vector<float> &values, float mean, float *stddevPtr) {
  int len = (int) values.size();
  if (len == 0 || len == 1) {
    *stddevPtr = 0.0f;
    return;
  }
  
  float sum = 0.0f;
  for (vector<float>::iterator it = values.begin(); it != values.end(); ++it) {
    float value = *it;
    float delta = value - mean;
    sum += (delta * delta);
  }
  if (sum == 0.0f) {
    *stddevPtr = 0.0f;
  } else {
    *stddevPtr = sqrt(sum / (len - 1));
  }
}

// Calculate diffs between float values and return as a vector.
// Return list of (N - (N-1)) elements in the list where the first element
// is always (values[0] - 0.0)

vector<float>
float_diffs(vector<float> &values) {
  const bool debug = false;
  
  if (debug) {
    cout << "float_diffs() for values" << endl;
    for (vector<float>::iterator it = values.begin(); it != values.end(); ++it) {
      float value = *it;
      cout << value << endl;
    }
  }
  
  float last = 0.0;
  
  vector<float> deltas;
  
  for (vector<float>::iterator it = values.begin(); it != values.end(); ++it) {
    float value = *it;
    float delta = value - last;
    deltas.push_back(delta);
    last = value;
  }
  
  if (debug) {
    cout << "returning deltas" << endl;
    for (vector<float>::iterator it = deltas.begin(); it != deltas.end(); ++it) {
      float delta = *it;
      cout << delta << endl;
    }
  }
  
  return deltas;
}

// Util method to return the 8 neighbors of a center point in the order
// R, U, L, D, UR, UL, DL, DR while taking the image bounds into
// account. For example, the point (0, 1) will not return UL, L, or DL.
// The iteration order here is implicitly returning all the pixels
// that are sqrt(dx,dy)=1 first and then the 1.4 distance pixels last.

const vector<Coord>
get8Neighbors(Coord center, int32_t width, int32_t height)
{
  int32_t cX = center.x;
  int32_t cY = center.y;
  
#if defined(DEBUG)
  assert(cX >= 0);
  assert(cX < width);
  assert(cY >= 0);
  assert(cY < height);
#endif // DEBUG

  static
  int32_t neighborOffsetsArr[] = {
    1,  0,  // R
    0, -1,  // U
    -1,  0, // L
    0,  1,  // D
    1, -1,  // UR
    -1, -1, // UL
    -1,  1, // DL
    1,  1   // DR
  };
  
  Coord neighbors[8];
  size_t neighborsOffset = 0;
  
  for (int i = 0; i < sizeof(neighborOffsetsArr)/sizeof(int32_t); i += 2) {
    int32_t dX = neighborOffsetsArr[i];
    int32_t dY = neighborOffsetsArr[i+1];
   
    int32_t X = cX + dX;
    int32_t Y = cY + dY;
    
    if (X < 0 || X >= width) {
      // Ignore this coordinate since it is outside bbox
    } else if (Y < 0 || Y >= height) {
      // Ignore this coordinate since it is outside bbox
    } else {
      Coord neighbor(X, Y);
#if defined(DEBUG)
      assert(neighborsOffset >= 0 && neighborsOffset < 8);
#endif // DEBUG
      neighbors[neighborsOffset++] = neighbor;
    }
  }
  
  // Construct vector object by reading values of known size from stack
  
  const auto it = begin(neighbors);
  vector<Coord> neighborsVec(it, it+neighborsOffset);
#if defined(DEBUG)
  assert(neighborsVec.size() == neighborsOffset);
#endif // DEBUG
  return neighborsVec;
}

static
bool CompareCoordIntWeightTupleDecreasingFunc (CoordIntWeightTuple &elem1, CoordIntWeightTuple &elem2) {
  int32_t weight1 = get<1>(elem1);
  int32_t weight2 = get<1>(elem2);
  return (weight1 > weight2);
}

static
bool CompareCoordIntWeightTupleIncreasingFunc (CoordIntWeightTuple &elem1, CoordIntWeightTuple &elem2) {
  int32_t weight1 = get<1>(elem1);
  int32_t weight2 = get<1>(elem2);
  return (weight1 < weight2);
}

void
sortCoordIntWeightTuples(vector<CoordIntWeightTuple> &tuples, bool decreasing)
{
  if (decreasing) {
    sort(tuples.begin(), tuples.end(), CompareCoordIntWeightTupleDecreasingFunc);
  } else {
    sort(tuples.begin(), tuples.end(), CompareCoordIntWeightTupleIncreasingFunc);
  }
}

// Given a binary image consisting of zero or non-zero values, calculate
// the center of mass for a shape defined by the white pixels.

void centerOfMass(uint8_t *bytePtr, uint32_t width, uint32_t height, uint32_t *xPtr, uint32_t *yPtr)
{
  uint32_t sumX = 0;
  uint32_t sumY = 0;
  uint32_t N = 0;
  
  uint32_t offset = 0;
  
  for (int y=0; y < height; y++) {
    for (int x=0; x < width; x++) {
      uint8_t val = bytePtr[offset];
      if (val) {
        sumX += x;
        sumY += y;
        N += 1;
      }
    }
  }
  
  *xPtr = sumX / N;
  *yPtr = sumY / N;
  return;
}

// my_adler32

// largest prime smaller than 65536
#define BASE 65521L

// NMAX is the largest n such that 255n(n+1)/2 + (n+1)(BASE-1) <= 2^32-1
#define NMAX 5552

#define DO1(buf, i)  { s1 += buf[i]; s2 += s1; }
#define DO2(buf, i)  DO1(buf, i); DO1(buf, i + 1);
#define DO4(buf, i)  DO2(buf, i); DO2(buf, i + 2);
#define DO8(buf, i)  DO4(buf, i); DO4(buf, i + 4);
#define DO16(buf)    DO8(buf, 0); DO8(buf, 8);

uint32_t my_adler32(
                    uint32_t adler,
                    unsigned char const *buf,
                    uint32_t len,
                    uint32_t singleCallMode)
{
  int k;
  uint32_t s1 = adler & 0xffff;
  uint32_t s2 = (adler >> 16) & 0xffff;
  
  if (!buf)
    return 1;
  
  while (len > 0) {
    k = len < NMAX ? len :NMAX;
    len -= k;
    while (k >= 16) {
      DO16(buf);
      buf += 16;
      k -= 16;
    }
    if (k != 0)
      do {
        s1 += *buf++;
        s2 += s1;
      } while (--k);
    s1 %= BASE;
    s2 %= BASE;
  }
  
  uint32_t result = (s2 << 16) | s1;
  
  if (singleCallMode && (result == 0)) {
    // All zero input, use 0xFFFFFFFF instead
    result = 0xFFFFFFFF;
  }
  
  return result;
}
