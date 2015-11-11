//
//  Coord.h
//  SeedsCombined
//
//  Created by Mo DeJong on 2/14/15.
//  Copyright (c) 2015 HelpURock Software. All rights reserved.
//
//  A coord represents a pair of (X, Y) values internally stored
//  a 16 bit unsigned integer values. These coord values operate
//  in the range 0 -> 0xFFFF (65535) which can represent
//  coordinates for even the largest 2D images as 32bits of data.
//  Since image data can contain a very large number of coords
//  this class reduces memory usage as compared to use of a
//  pair<int, int> which can consume 64 or or even 128 bits of mem.
//  This coordinate object can also be used in an unordered map
//  as a unique key.

#ifndef __Superpixel__Coord__
#define __Superpixel__Coord__

#include <unistd.h>
#include <stdlib.h>
#include <assert.h>

#include <string>
#include <ostream>

using namespace std;

class Coord {
public:
  uint16_t x;
  uint16_t y;

  Coord()
  :x(0), y(0)
  {
  }
  
  Coord(uint16_t X, uint16_t Y)
  :x(X), y(Y)
  {
  }
  
  Coord(uint32_t X, uint32_t Y)
    :x((uint16_t)X), y((uint16_t)Y)
  {
  }
  
  Coord(int X, int Y)
  {
    assert(X >= 0 && X <= 0xFFFF);
    x = (uint16_t) X;
    assert(Y >= 0 && Y <= 0xFFFF);
    y = (uint16_t) Y;
  }
  
  // Calculate generic X/Y offset to enable single value compare
  // in terms of rows and then columns.
  
  uint32_t calcOffset() const {
    uint32_t offset = (y * 0xFFFF) + x;
    return offset;
  }
  
  // Calculate offset for a specific width, this implementation
  // is slower than calcOffset() and should be used when
  // looking up an offset in a real memory buffer.
  
  uint32_t offsetFor(uint32_t width) const {
    uint32_t offset = (y * width) + x;
    return offset;
  }
  
  bool operator==(const Coord &other) const {
    return (x == other.x && y == other.y);
  }
  
  bool operator!=(const Coord &other) const {
    return !(*this == other);
  }
  
  bool operator<(const Coord &other) const
  {
    return calcOffset() < other.calcOffset();
  }

  bool operator<=(const Coord &other) const
  {
    return calcOffset() <= other.calcOffset();
  }
  
  bool operator>(const Coord &other) const
  {
    return calcOffset() > other.calcOffset();
  }
  
  bool operator>=(const Coord &other) const
  {
    return calcOffset() >= other.calcOffset();
  }
  
  bool operator()(const Coord &lhs, const Coord &rhs) const
  {
    return lhs == rhs;
  }
  
  size_t gethash() const
  {
    size_t hashed;
    size_t h1 = x;
    size_t h2 = y;
    hashed = h1 | (h2 << 16);
    return hashed;
  }
  
  // Format the coord as a string with fixed width output for each tag
  
  string toString() const {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "(%d, %d)", x, y);
    return string(buffer);
  }
  
  // Enable writing coord directly to stream via overloading
  
  friend ostream& operator<<(ostream& os, const Coord& coord) {
    os << coord.toString();
    return os;
  }
};

// support hash function so that Coord can be an unordered map key

namespace std {
  template <>
  class hash<Coord>{
    public :
    size_t operator()(const Coord &coord) const
    {
      return coord.gethash();
    }
  };
};

#endif /* defined(__Superpixel__Coord__) */
