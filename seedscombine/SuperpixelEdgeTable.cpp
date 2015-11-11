// An edge table represents edges as constant time lookup tables that make it possible to
// iterate over an update graph nodes and edges in constant time.

#include "Superpixel.h"

#include "SuperpixelEdgeTable.h"

SuperpixelEdgeTable::SuperpixelEdgeTable()
{
}

// This implementation iterates over all the edges in the graph and it is not fast
// for a graph with many nodes.

vector<SuperpixelEdge> SuperpixelEdgeTable::getAllEdges()
{
  const bool debug = false;
  
  // Walk over all superpixels and get each neighbor, then create an edge only when the UID
  // of a superpixel is smaller than the UID of the edge. This basically does a dedup of
  // all the edges without creating and checking for dups.
  
  vector<SuperpixelEdge> allEdges;
  
  vector<int32_t> allSuperpixles = getAllTagsInNeighborsTable();
  
  for (vector<int32_t>::iterator it = allSuperpixles.begin(); it != allSuperpixles.end(); ++it ) {
    int32_t tag = *it;
    
    vector<int32_t> neighbors = getNeighbors(tag);
    
    if (debug) {
      cout << "for superpixel " << tag << " neighbors:" << endl;
    }
    
    for (vector<int32_t>::iterator neighborIter = neighbors.begin(); neighborIter != neighbors.end(); ++neighborIter ) {
      int32_t neighborTag = *neighborIter;

      if (debug) {
        cout << neighborTag << endl;
      }
      
      if (tag <= neighborTag) {
        SuperpixelEdge edge(tag, neighborTag);
        allEdges.push_back(edge);
        
        if (debug) {
          cout << "added edge (" << edge.A << "," << edge.B << ")" << endl;
        }
      } else {
        if (debug) {
          SuperpixelEdge edge(tag, neighborTag);
          
          cout << "ignored dup edge (" << edge.A << "," << edge.B << ")" << endl;
        }
      }
    }
  }
  
  return allEdges;
}

// Optimal getNeighborsPtr() returns a pointer into the table which is
// fast WRT iteration. The caller has to very careful not to attempt
// to iterate over an invocation of merge edge since that can change
// the neighbor list.

vector<int32_t>* SuperpixelEdgeTable::getNeighborsPtr(int32_t tag)
{
  //assert(neighbors.count(tag) > 0);
  //return neighbors[tag];
  
  unordered_map <int32_t, vector<int32_t> >::iterator iter = neighbors.find(tag);
  
  if (iter == neighbors.end()) {
    // Neighbors key must be defined for this tag
    assert(0);
  } else {
    // Otherwise the key exists in the table, return ref to vector in table
    // with the assumption that the caller will not change it.
    
    return &iter->second;
  }
}

// This impl returns a copy of the neighbors vector

vector<int32_t> SuperpixelEdgeTable::getNeighbors(int32_t tag)
{
  return *getNeighborsPtr(tag);
}

// Set initial list of neighbors for a superpixel or rest the list after making
// changes. The neighbor values are sorted only to make the results easier to
// read, there should not be much impact on performance since the list of neighbors
// is typically small.

void SuperpixelEdgeTable::setNeighbors(int32_t tag, vector<int32_t> neighborsUIDsVec)
{
  sort (neighborsUIDsVec.begin(), neighborsUIDsVec.end());
  neighbors[tag] = neighborsUIDsVec;
}

// When deleting a node, remove the neighbor entries

void SuperpixelEdgeTable::removeNeighbors(int32_t tag)
{
  neighbors.erase(tag);
}

// Return a vector of tags that have an entry in the neighbors table.
// Even if the vector contains zero elements, this method is not fast.

vector<int32_t> SuperpixelEdgeTable::getAllTagsInNeighborsTable()
{
  vector<int32_t> vec;
  for ( unordered_map <int32_t, vector<int32_t> >::iterator it = neighbors.begin(); it != neighbors.end(); ++it ) {
    vec.push_back(it->first);
  }
  sort (vec.begin(), vec.end());
  return vec;
}
