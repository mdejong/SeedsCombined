// A superpixel edge represents a connection between superpixels that are located next to each other.
// An edge contains only UID values that correspond to superpixels defined in a superpixel image.

#ifndef SUPERPIXEL_EDGE_H
#define	SUPERPIXEL_EDGE_H

#include <opencv2/opencv.hpp>

class Superpixel;

using namespace std;
using namespace cv;

class SuperpixelEdge {
  
  public:
  
  SuperpixelEdge(int32_t uid1, int32_t uid2);
  
  int32_t A;
  int32_t B;
  
  bool operator==(const SuperpixelEdge &other) const {
    return (A == other.A && B == other.B);
  }
  
  bool operator()(const SuperpixelEdge &lhs, const SuperpixelEdge &rhs) const
  {
    return lhs == rhs;
  }
  
  size_t gethash() const
  {
    size_t hashed;
    if (sizeof(size_t) == sizeof(uint32_t)) {
      // On a 32bit arch use the lower 16bits of h1 and h2
      size_t h1 = A & 0x0000FFFF;
      size_t h2 = B & 0x0000FFFF;
      hashed = h1 | (h2 << 16);
    } else {
      // On a 64bit arch join the two 32 bit numbers as
      // a single 64bit unsigned hash.
      size_t h1 = A & 0x00FFFFFF;
      size_t h2 = B & 0x00FFFFFF;
      hashed = h1 | (h2 << 32);
    }
    return hashed;
  }
  
  // Format the edge as a string with fixed width output for each tag
  
  string toString() const {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "(%d, %d)", A, B);
    return string(buffer);
  }
  
  // Enable writing edge directly to stream via overloading
  
  friend ostream& operator<<(ostream& os, const SuperpixelEdge& edge) {
    os << edge.toString();
    return os;
  }
};

// support hash function so that SuperPixelEdge can be an unordered map key

namespace std {
  template <>
  class hash<SuperpixelEdge>{
    public :
    size_t operator()(const SuperpixelEdge &edge) const
    {
      return edge.gethash();
    }
  };
};

#endif // SUPERPIXEL_EDGE_H
