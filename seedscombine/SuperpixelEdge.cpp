// A superpixel edge represents a connection between superpixels that are located next to each other.

#include "Superpixel.h"

#include "SuperpixelEdge.h"

SuperpixelEdge::SuperpixelEdge(int32_t uid1, int32_t uid2)
{
  // Construct efge (A, B) where the order is always known as (min, max)
  // so that constructing another edge object will 
  
  if (uid1 <= uid2) {
    A = uid1;
    B = uid2;
  } else {
    A = uid2;
    B = uid1;
  }
}
