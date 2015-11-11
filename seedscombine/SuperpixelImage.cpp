// A superpixel image is a matrix that contains N superpixels and N superpixel edges between superpixels.
// A superpixel image is typically parsed from a source of tags, modified, and then written as a new tags
// image.

#include "SuperpixelImage.h"

#include "Superpixel.h"

#include "SuperpixelEdge.h"

#include "SuperpixelEdgeTable.h"

#include "Util.h"

#include "OpenCVUtil.h"

#include <iomanip>      // setprecision

const int MaxSmallNumPixelsVal = 10;

void parse3DHistogram(Mat *histInputPtr,
                      Mat *histPtr,
                      Mat *backProjectInputPtr,
                      Mat *backProjectPtr,
                      int conversion,
                      int numBins);

void writeTagsWithStaticColortable(SuperpixelImage &spImage, Mat &resultImg);

bool pos_sample_within_bound(vector<float> &weights, float currentWeight);

void writeSuperpixelMergeMask(SuperpixelImage &spImage, Mat &resultImg, vector<int32_t> merges, vector<float> weights, unordered_map<int32_t, bool> *lockedTablePtr);

void generateStaticColortable(Mat &inputImg, SuperpixelImage &spImage);

void writeTagsWithStaticColortable(SuperpixelImage &spImage, Mat &resultImg);

// Note that the valid range for tags is 0 -> 0x00FFFFFF so that
// -1 can be used to indicate no tag. There will never be so many
// tags that 24bits is not enough tags.

typedef unordered_map<int32_t, vector<int32_t> > TagToNeighborMap;

// Sort by size with largest superpixel first

typedef
struct SuperpixelSortStruct
{
  Superpixel *spPtr;
} SuperpixelSortStruct;

static
bool CompareSuperpixelSizeDecreasingFunc (SuperpixelSortStruct &s1, SuperpixelSortStruct &s2) {
  Superpixel *sp1Ptr = s1.spPtr;
  Superpixel *sp2Ptr = s2.spPtr;
  int sp1N = (int) sp1Ptr->coords.size();
  int sp2N = (int) sp2Ptr->coords.size();
  if (sp1N == sp2N) {
    // In case of a tie, sort by tag so that edges are processed in
    // increasing tag order to help with edge auto ordering.
    int32_t tag1 = sp1Ptr->tag;
    int32_t tag2 = sp2Ptr->tag;
    return (tag1 < tag2);
  } else {
    return (sp1N > sp2N);
  }
}

bool SuperpixelImage::parse(Mat &tags, SuperpixelImage &spImage) {
  const bool debug = false;
  
  assert(tags.channels() == 3);
  
  TagToSuperpixelMap &tagToSuperpixelMap = spImage.tagToSuperpixelMap;
  
  for( int y = 0; y < tags.rows; y++ ) {
    for( int x = 0; x < tags.cols; x++ ) {
      Vec3b tagVec = tags.at<Vec3b>(y, x);
      int32_t tag = Vec3BToUID(tagVec);
      
      // Note that each tag value is modified here so that no superpixel
      // will have the tag zero.
      
      if (1) {
        // Note that an input tag value must always be smaller than 0x00FFFFFF
        // since this logic will implicitly add 1 to each pixel value to make
        // sure that zero is not used as a valid tag value while processing.
        // This means that the image cannot use the value for all white as
        // a valid tag value, but that is not a big deal since every other value
        // can be used.
        
        if (tag == 0xFFFFFF) {
          cerr << "error : tag pixel has the value 0xFFFFFF which is not supported" << endl;
          return false;
        }
        assert(tag < 0x00FFFFFF);
        tag += 1;
        tagVec[0] = tag & 0xFF;
        tagVec[1] = (tag >> 8) & 0xFF;
        tagVec[2] = (tag >> 16) & 0xFF;
        tags.at<Vec3b>(y, x) = tagVec;
      }
      
      TagToSuperpixelMap::iterator iter = tagToSuperpixelMap.find(tag);
      
      if (iter == tagToSuperpixelMap.end()) {
        // A Superpixel has not been created for this UID since no key
        // exists in the table. Create a superpixel and wrap into a
        // unique smart pointer so that the table contains the only
        // live object reference to the Superpixel object.
        
        if (debug) {
          cout << "create Superpixel for UID " << tag << endl;
        }
        
        Superpixel *spPtr = new Superpixel(tag);
        iter = tagToSuperpixelMap.insert(iter, make_pair(tag, spPtr));
      } else {
        if (debug) {
          cout << "exists  Superpixel for UID " << tag << endl;
        }
      }

      Superpixel *spPtr = iter->second;
      assert(spPtr->tag == tag);
      
      spPtr->appendCoord(x, y);
    }
  }
  
  // Collect all superpixels as a single vector sorted by increasing UID values
  
  vector<int32_t> &superpixels = spImage.superpixels;
  
  for (TagToSuperpixelMap::iterator it = tagToSuperpixelMap.begin(); it!=tagToSuperpixelMap.end(); ++it) {
    //int32_t tag = it->first;
    Superpixel *spPtr = it->second;
    superpixels.push_back(spPtr->tag);
  }

  sort (superpixels.begin(), superpixels.end());
  
  // Print superpixel info
  
  if (debug) {
    cout << "added " << (tags.rows * tags.cols) << " pixels as " << superpixels.size() << " superpixels" << endl;
    
    for (vector<int32_t>::iterator it = superpixels.begin(); it != superpixels.end(); ++it) {
      int32_t tag = *it;
      
      assert(tagToSuperpixelMap.count(tag) > 0);
      Superpixel *spPtr = tagToSuperpixelMap[tag];
      
      cout << "superpixel UID = " << tag << " contains " << spPtr->coords.size() << " coords" << endl;
      
      for (vector<pair<int32_t,int32_t> >::iterator coordsIt = spPtr->coords.begin(); coordsIt != spPtr->coords.end(); ++coordsIt) {
        int32_t X = coordsIt->first;
        int32_t Y = coordsIt->second;
        cout << "X,Y" << " " << X << "," << Y << endl;
      }
    }
  }
  
  // Generate edges for each superpixel by looking at superpixel UID's around a given X,Y coordinate
  // and determining the other superpixels that are connected to each superpixel.
  
  bool worked = SuperpixelImage::parseSuperpixelEdges(tags, spImage);
  
  if (!worked) {
    return false;
  }
  
  // Deallocate original input tags image since it could be quite large
  
  //tags.release();
  
  return true;
}

// Examine superpixels in an image and parse edges from the superpixel coords

bool SuperpixelImage::parseSuperpixelEdges(Mat &tags, SuperpixelImage &spImage) {
  const bool debug = false;
  
  vector<int32_t> &superpixels = spImage.superpixels;
  
  //vector<SuperpixelEdge> &edges = spImage.edges;
  
  SuperpixelEdgeTable &edgeTable = spImage.edgeTable;
  
  // For each (X,Y) tag value find all the other tags defined in suppounding 8 pixels
  // and dedup the list of neighbor tags for each superpixel.

  int32_t neighborOffsetsArr[] = {
    -1, -1, // UL
     0, -1, // U
     1, -1, // UR
    -1,  0, // L
     1,  0, // R
    -1,  1, // DL
     0,  1, // D
     1,  1  // DR
  };
  
  vector<pair<int32_t, int32_t> > neighborOffsets;
  
  for (int i = 0; i < sizeof(neighborOffsetsArr)/sizeof(int32_t); i += 2) {
    pair<int32_t,int32_t> p(neighborOffsetsArr[i], neighborOffsetsArr[i+1]);
    neighborOffsets.push_back(p);
  }
  
  assert(neighborOffsets.size() == 8);
  
  TagToNeighborMap tagToNeighborMap;
  
  for( int y = 0; y < tags.rows; y++ ) {
    for( int x = 0; x < tags.cols; x++ ) {
      Vec3b tagVec = tags.at<Vec3b>(y, x);
      
      int32_t centerTag = Vec3BToUID(tagVec);
      
      if (debug) {
      cout << "center (" << x << "," << y << ") with tag " << centerTag << endl;
      }
      
      TagToNeighborMap::iterator iter = tagToNeighborMap.find(centerTag);
      
      if (iter == tagToNeighborMap.end()) {
        // A Superpixel has not been created for this UID since no key
        // exists in the table.
        
        if (debug) {
          cout << "create neighbor vector for UID " << centerTag << endl;
        }
        
        vector<int32_t> neighborVec;
        iter = tagToNeighborMap.insert(iter, make_pair(centerTag, neighborVec));
      } else {
        if (debug) {
          cout << "exits  neighbor vector for UID " << centerTag << endl;
        }
      }
      
      vector<int32_t> &neighborUIDsVec = iter->second;

      // Loop over each neighbor around (X,Y) and lookup tag
      
      for (vector<pair<int32_t, int32_t>>::iterator pairIter = neighborOffsets.begin() ; pairIter != neighborOffsets.end(); ++pairIter) {
        int dX = pairIter->first;
        int dY = pairIter->second;
        
        int foundNeighborUID;
        
        int nX = x + dX;
        int nY = y + dY;
        
        if (nX < 0 || nX >= tags.cols) {
          foundNeighborUID = -1;
        } else if (nY < 0 || nY >= tags.rows) {
          foundNeighborUID = -1;
        } else {
          Vec3b neighborTagVec = tags.at<Vec3b>(nY, nX);
          foundNeighborUID = Vec3BToUID(neighborTagVec);
        }

        if (foundNeighborUID == -1 || foundNeighborUID == centerTag) {
          if (debug) {
            cout << "ignoring (" << nX << "," << nY << ") with tag " << foundNeighborUID << " since invalid or identity" << endl;
          }
        } else {
          bool found = false;

          if (debug) {
          cout << "checking (" << nX << "," << nY << ") with tag " << foundNeighborUID << " to see if known neighbor" << endl;
          }
          
          for (vector<int>::iterator it = neighborUIDsVec.begin() ; it != neighborUIDsVec.end(); ++it) {
            int32_t knownNeighborUID = *it;
            if (foundNeighborUID == knownNeighborUID) {
              // This collection of neighbors already contains an entry for this neighbor
              found = true;
              break;
            }
          }
          
          if (!found) {
            neighborUIDsVec.push_back(foundNeighborUID);
            
            if (debug) {
            cout << "added new neighbor tag " << foundNeighborUID << endl;
            }
          }
        }
      }
      
      if (debug) {
      cout << "after searching all neighbors of (" << x << "," << y << ") the neighbors array (len " << neighborUIDsVec.size() << ") is:" << endl;
      
      for (vector<int>::iterator it = neighborUIDsVec.begin() ; it != neighborUIDsVec.end(); ++it) {
        int32_t knownNeighborUID = *it;
        cout << knownNeighborUID << endl;
      }
      }
      
    }
  }
  
  // Each superpixel now has a vector of values for each neighbor. Create unique list of edges by
  // iterating over the superpixels and only creating an edge object when a pair (A, B) is
  // found where A < B.
  
  for (vector<int32_t>::iterator it = superpixels.begin(); it != superpixels.end(); ++it) {
    int32_t tag = *it;
    
    // Foreach superpixel, collect all the connected neighbors and generate edges
    
#if defined(DEBUG)
    assert(tagToNeighborMap.count(tag) > 0);
#endif // DEBUG
    
    vector<int32_t> &neighborUIDsVec = tagToNeighborMap[tag];

    if (debug) {
    cout << "superpixel UID = " << tag << " neighbors len = " << neighborUIDsVec.size() << endl;
    }
    
    // Every superpixel must have at least 1 neighbor or the input is invalid, unless there
    // is only 1 superpixel to begin with which could happen if all input was same pixel.
    
    if (superpixels.size() > 1) {
      assert(neighborUIDsVec.size() > 0);
    }
    
    edgeTable.setNeighbors(tag, neighborUIDsVec);
  }

  if (debug) {
    cout << "created " << edgeTable.getAllEdges().size() << " edges in edge table" << endl;
  }

  return true;
}

void SuperpixelImage::mergeEdge(SuperpixelEdge &edgeToMerge) {
  const bool debug = false;
  
  if (debug) {
    cout << "mergeEdge (" << edgeToMerge.A << " " << edgeToMerge.B << ")" << endl;
  }
  
  assert(edgeToMerge.A != edgeToMerge.B);
  
#if defined(DEBUG)
  mergeOrder.push_back(edgeToMerge);
#endif
  
  // Get Superpixel object pointers (not copies of the objects)
  
  Superpixel *spAPtr = getSuperpixelPtr(edgeToMerge.A);
  assert(spAPtr);
  Superpixel *spBPtr = getSuperpixelPtr(edgeToMerge.B);
  assert(spBPtr);

  Superpixel *srcPtr;
  Superpixel *dstPtr;
  
  size_t numCoordsA;
  size_t numCoordsB;
  
  numCoordsA = spAPtr->coords.size();
  numCoordsB = spBPtr->coords.size();
  
  if (numCoordsA >= numCoordsB) {
    // Merged B into A since A is larger

    srcPtr = spBPtr;
    dstPtr = spAPtr;
    
    if (debug) {
      cout << "merge B -> A : " << srcPtr->tag << " -> " << dstPtr->tag << " : " << srcPtr->coords.size() << " <= " << dstPtr->coords.size() << endl;
    }
  } else {
    // Merge A into B since B is larger
    
    srcPtr = spAPtr;
    dstPtr = spBPtr;
    
    if (debug) {
      cout << "merge A -> B : " << srcPtr->tag << " -> " << dstPtr->tag << " : " << srcPtr->coords.size() << " < " << dstPtr->coords.size() << endl;
    }
  }
  
  if (debug) {
    cout << "will merge " << srcPtr->coords.size() << " coords from smaller into larger superpixel" << endl;
  }

  for (vector <pair<int32_t,int32_t> >::iterator it = srcPtr->coords.begin(); it != srcPtr->coords.end(); ++it) {
    dstPtr->coords.push_back(*it);
  }
  
  srcPtr->coords.resize(0);
  
  // Find entry for srcPtr->tags in superpixels and remove the UID
  
  bool found = false;
  
  for (vector<int32_t>::iterator it = superpixels.begin(); it != superpixels.end(); ++it) {
    int32_t tag = *it;
    
    if (srcPtr->tag == tag) {
      if (debug) {
        cout << "superpixel UID = " << tag << " found as delete match in ordered superpixels list" << endl;
      }
      
      it = superpixels.erase(it);
      found = true;
      break;
    }
  }
  
  assert(found);
  
  // Remove edge between src and dst by removing src from dst neighbors
  
  vector<int32_t> neighborsDst = edgeTable.getNeighbors(dstPtr->tag);
  
  found = false;
  
  for (vector<int32_t>::iterator it = neighborsDst.begin(); it != neighborsDst.end(); ) {
    int32_t neighborOfDstTag = *it;

    if (debug) {
      cout << "iter neighbor of dst = " << neighborOfDstTag << endl;
    }
    
    if (neighborOfDstTag == srcPtr->tag) {
      if (debug) {
        cout << "remove src from dst neighbor list" << endl;
      }
      
      it = neighborsDst.erase(it);
      found = true;
    } else {
      ++it;
    }
    
    // Clear edge strength cache of any edge that involves dst
    
    SuperpixelEdge cachedKey(dstPtr->tag, neighborOfDstTag);
    edgeTable.edgeStrengthMap.erase(cachedKey);
  }
  
  assert(found);
  
  edgeTable.setNeighbors(dstPtr->tag, neighborsDst);
  
  // Update all neighbors of src by removing src as a neighbor
  // and then adding dst if it does not already exist.
  
  vector<int32_t> neighborsSrc = edgeTable.getNeighbors(srcPtr->tag);
  
  found = false;
  
  for (vector<int32_t>::iterator it = neighborsSrc.begin(); it != neighborsSrc.end(); ++it ) {
    int32_t neighborOfSrcTag = *it;
    
    if (debug) {
      cout << "iter neighbor of src = " << neighborOfSrcTag << endl;
    }
    
    // Clear edge strength cache of any edge that involves src
    
    SuperpixelEdge cachedKey(srcPtr->tag, neighborOfSrcTag);
    edgeTable.edgeStrengthMap.erase(cachedKey);
    
    if (neighborOfSrcTag == dstPtr->tag) {
      // Ignore dst so that src is deleted as a neighbor
      
      if (debug) {
        cout << "ignore neighbor of src since it is the dst node" << endl;
      }
    } else {
      unordered_map<int32_t,int32_t> neighborsOfSrcNotDstMap;
      
      vector<int32_t> neighbors = edgeTable.getNeighbors(neighborOfSrcTag);
      
      found = false;
      
      for (vector<int32_t>::iterator neighborIter = neighbors.begin(); neighborIter != neighbors.end(); ) {
        int32_t neighborOfSrcTag = *neighborIter;
        
        if (neighborOfSrcTag == srcPtr->tag) {
          if (debug) {
            cout << "remove src from src neighbor list" << endl;
          }
          
          neighborIter = neighbors.erase(neighborIter);
          found = true;
        } else {
          neighborsOfSrcNotDstMap[neighborOfSrcTag] = 0;
          ++neighborIter;
        }
      }
      
      assert(found);
      
      if (debug) {
        cout << "neighborsOfSrcNotDstMap size() " << neighborsOfSrcNotDstMap.size() << " for neighbor of src " << neighborOfSrcTag << endl;
        
        for ( unordered_map <int32_t,int32_t>::iterator it = neighborsOfSrcNotDstMap.begin(); it != neighborsOfSrcNotDstMap.end(); ++it ) {
          cout << "neighborsOfSrcNotDstMap[" << it->first << "]" << endl;
        }
      }
      
      if (neighborsOfSrcNotDstMap.count(dstPtr->tag) == 0) {
        // dst is not currently a neighbor of this neighbor of src, make it one now
        neighbors.push_back(dstPtr->tag);
        
        if (debug) {
          cout << "added dst to neighbor of src node " << neighborOfSrcTag << endl;
        }
        
        vector<int32_t> neighborsDst = edgeTable.getNeighbors(dstPtr->tag);
        neighborsDst.push_back(neighborOfSrcTag);
        edgeTable.setNeighbors(dstPtr->tag, neighborsDst);
        
        if (debug) {
          cout << "added neighbor of src node to dst neighbors" << endl;
        }
      }
      
      edgeTable.setNeighbors(neighborOfSrcTag, neighbors);
    }
  }
  
  edgeTable.removeNeighbors(srcPtr->tag);
  
  // Move edge weights from src to dst
  
  if (srcPtr->mergedEdgeWeights.size() > 0) {
    append_to_vector(dstPtr->mergedEdgeWeights, srcPtr->mergedEdgeWeights);
  }

  if (srcPtr->unmergedEdgeWeights.size() > 0) {
    append_to_vector(dstPtr->unmergedEdgeWeights, srcPtr->unmergedEdgeWeights);
  }
  
  // Finally remove the Superpixel object from the lookup table and free the memory
  
  int32_t tagToRemove = srcPtr->tag;
  tagToSuperpixelMap.erase(tagToRemove);
  delete srcPtr;
  
#if defined(DEBUG)
  // When compiled in DEBUG mode in Xcode enable additional runtime checks that
  // ensure that each neighbor of the merged node is also a neighbor of the other.

  srcPtr = getSuperpixelPtr(tagToRemove);
  assert(srcPtr == NULL);
  dstPtr = getSuperpixelPtr(dstPtr->tag);
  assert(dstPtr != NULL);
  
  vector<int32_t> *neighborsPtr = edgeTable.getNeighborsPtr(dstPtr->tag);
  
  for (vector<int32_t>::iterator neighborIter = neighborsPtr->begin(); neighborIter != neighborsPtr->end(); ++neighborIter ) {
    int32_t neighborTag = *neighborIter;
    
    // Make sure that each neighbor of the merged superpixel also has the merged superpixel
    // as a neighbor.
    
    Superpixel *neighborPtr = getSuperpixelPtr(neighborTag);
    assert(neighborPtr != NULL);
    
    vector<int32_t> *neighborsOfNeighborPtr = edgeTable.getNeighborsPtr(neighborTag);
    
    found = false;
    
    for (vector<int32_t>::iterator nnIter = neighborsOfNeighborPtr->begin(); nnIter != neighborsOfNeighborPtr->end(); ++nnIter ) {
      int32_t nnTag = *nnIter;
      if (nnTag == dstPtr->tag) {
        found = true;
        break;
      }
    }
    
    assert(found);
  }
  
  // Check that merge src no longer appers in superpixels list
  
  for (vector<int32_t>::iterator it = superpixels.begin(); it != superpixels.end(); ++it) {
    int32_t tag = *it;
    
    if (tagToRemove == tag) {
      assert(0);
    }
  }
#endif // DEBUG
  
  return;
}

// Lookup Superpixel* given a UID, checks to make sure key is defined in table in DEBUG mode

Superpixel* SuperpixelImage::getSuperpixelPtr(int32_t uid)
{
  TagToSuperpixelMap::iterator iter = tagToSuperpixelMap.find(uid);
  
  if (iter == tagToSuperpixelMap.end()) {
    // In the event that a superpixel was merged into a neighbor during an iteration
    // over a list of known superpixel then this method could be invoked with a uid
    // that is no longer valid. Return NULL to make it possible to detect this case.
    
    return NULL;
  } else {
    // Otherwise the key exists in the table, return the cached pointer to
    // avoid a second lookup because this method is invoked a lot.
    
    return iter->second;
  }
}

// Compare method for CompareNeighborTuple type, in the case of a tie the second column
// is sorted in terms of decreasing int values.

static
bool CompareNeighborTupleFunc (CompareNeighborTuple &elem1, CompareNeighborTuple &elem2) {
  double hcmp1 = get<0>(elem1);
  double hcmp2 = get<0>(elem2);
  if (hcmp1 == hcmp2) {
    int numPixels1 = get<1>(elem1);
    int numPixels2 = get<1>(elem2);
    return (numPixels1 > numPixels2);
  }
  return (hcmp1 < hcmp2);
}

// Sort tuple (UNUSED, UID, SIZE) by decreasing SIZE values

static
bool CompareNeighborTupleSortByDecreasingLargestNumCoordsFunc (CompareNeighborTuple &elem1, CompareNeighborTuple &elem2) {
  int numPixels1 = get<2>(elem1);
  int numPixels2 = get<2>(elem2);
  return (numPixels1 > numPixels2);
}

// Sort into decreasing order in terms of the float value in the first element of the tuple.
// In the case of a tie then sort by decreasing superpixel size.

static
bool CompareNeighborTupleDecreasingFunc (CompareNeighborTuple &elem1, CompareNeighborTuple &elem2) {
  double hcmp1 = get<0>(elem1);
  double hcmp2 = get<0>(elem2);
  if (hcmp1 == hcmp2) {
    int numPixels1 = get<1>(elem1);
    int numPixels2 = get<1>(elem2);
    return (numPixels1 > numPixels2);
  }
  return (hcmp1 > hcmp2);
}


// This method is invoked with a superpixel tag to generate a vector of tuples that compares
// the superpixel to all of the neighbor superpixels.
//
// TUPLE (BHATTACHARYYA N_PIXELS NEIGHBOR_TAG)
// BHATTACHARYYA : double
// N_PIXELS      : int32_t
// NEIGHBOR_TAG  : int32_t

void
SuperpixelImage::compareNeighborSuperpixels(Mat &inputImg,
                                            int32_t tag,
                                            vector<CompareNeighborTuple> &results,
                                            unordered_map<int32_t, bool> *lockedTablePtr,
                                            int32_t step) {
  const bool debug = false;
  const bool debugShowSorted = false;
  const bool debugDumpSuperpixels = false;
  
  vector<int32_t> *neighborsPtr = edgeTable.getNeighborsPtr(tag);
  
  Mat srcSuperpixelMat;
  Mat srcSuperpixelHist;
  Mat srcSuperpixelBackProjection;
  
  // Read RGB pixel data from main image into matrix for this one superpixel and then gen histogram.
  
  fillMatrixFromCoords(inputImg, tag, srcSuperpixelMat);
  
  parse3DHistogram(&srcSuperpixelMat, &srcSuperpixelHist, NULL, NULL, 0, -1);
  
  if (debugDumpSuperpixels) {
    std::ostringstream stringStream;
    if (step == -1) {
      stringStream << "superpixel_" << tag << ".png";
    } else {
      stringStream << "superpixel_step_" << step << "_" << tag << ".png";
    }
    std::string str = stringStream.str();
    const char *filename = str.c_str();
    
    cout << "write " << filename << " ( " << srcSuperpixelMat.cols << " x " << srcSuperpixelMat.rows << " )" << endl;
    imwrite(filename, srcSuperpixelMat);
  }
  
  if (!results.empty()) {
    results.erase (results.begin(), results.end());
  }
  
  for (vector<int32_t>::iterator neighborIter = neighborsPtr->begin(); neighborIter != neighborsPtr->end(); ++neighborIter) {
    // Generate histogram for the neighbor and then compare to neighbor
    
    int32_t neighborTag = *neighborIter;
    
    if (lockedTablePtr && (lockedTablePtr->count(neighborTag) != 0)) {
      // If a locked down table is provided then do not consider a neighbor that appears
      // in the locked table.
      
      if (debug) {
        cout << "skipping consideration of locked neighbor " << neighborTag << endl;
      }
      
      continue;
    }
    
    Mat neighborSuperpixelMat;
    Mat neighborSuperpixelHist;
    Mat neighborBackProjection;
    
    fillMatrixFromCoords(inputImg, neighborTag, neighborSuperpixelMat);
    
    parse3DHistogram(&neighborSuperpixelMat, &neighborSuperpixelHist, NULL, NULL, 0, -1);
    
    if (debugDumpSuperpixels) {
      std::ostringstream stringStream;
      stringStream << "superpixel_" << neighborTag << ".png";
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << neighborSuperpixelMat.cols << " x " << neighborSuperpixelMat.rows << " )" << endl;
      imwrite(filename, neighborSuperpixelMat);
    }
    
    assert(srcSuperpixelHist.dims == neighborSuperpixelHist.dims);
    
    double compar_bh = cv::compareHist(srcSuperpixelHist, neighborSuperpixelHist, CV_COMP_BHATTACHARYYA);
    
    if (debug) {
    cout << "BHATTACHARYYA " << compar_bh << endl;
    }
    
    CompareNeighborTuple tuple = make_tuple(compar_bh, neighborSuperpixelMat.cols, neighborTag);
    
    results.push_back(tuple);
  }
  
  if (debug) {
    cout << "unsorted tuples from src superpixel " << tag << endl;
    
    for (vector<CompareNeighborTuple>::iterator it = results.begin(); it != results.end(); ++it) {
      CompareNeighborTuple tuple = *it;
      char buffer[1024];
      snprintf(buffer, sizeof(buffer), "(%12.4f, %5d, %5d)",
               get<0>(tuple), get<1>(tuple), get<2>(tuple));
      cout << (char*)buffer << endl;
    }
  }
  
  // Sort tuples by BHATTACHARYYA value
  
  if (results.size() > 1) {
    sort(results.begin(), results.end(), CompareNeighborTupleFunc);
  }
  
  if (debug || debugShowSorted) {
    cout << "sorted tuples from src superpixel " << tag << endl;

    for (vector<CompareNeighborTuple>::iterator it = results.begin(); it != results.end(); ++it) {
      CompareNeighborTuple tuple = *it;
      char buffer[1024];
      snprintf(buffer, sizeof(buffer), "(%12.4f, %5d, %5d)",
               get<0>(tuple), get<1>(tuple), get<2>(tuple));
      cout << (char*)buffer << endl;
    }
  }
  
  return;
}

// This compare method will examine the pixels on an edge between a superpixel and each neighbor
// and then the neighbors will be returned in sorted order from smallest to largest in terms
// of the color diff between nearest pixels returned in a tuple of (NORMALIZE_DIST NUM_PIXELS TAG).
// Color compare operations are done in the LAB colorspace. The histogram compare looks at all
// the pixels in two superpixels, but this method is more useful when the logic wants to look
// at the values at the edges as opposed to the whole superpixel.

void
SuperpixelImage::compareNeighborEdges(Mat &inputImg,
                                      int32_t tag,
                                      vector<CompareNeighborTuple> &results,
                                      unordered_map<int32_t, bool> *lockedTablePtr,
                                      int32_t step,
                                      bool normalize) {
  const bool debug = false;
  const bool debugShowSorted = false;
  const bool debugDumpSuperpixelEdges = false;
  
  vector<int32_t> *neighborsPtr = edgeTable.getNeighborsPtr(tag);
   
  if (!results.empty()) {
    results.erase (results.begin(), results.end());
  }
  
  Superpixel *srcSpPtr = getSuperpixelPtr(tag);
  assert(srcSpPtr);
  
  for (vector<int32_t>::iterator neighborIter = neighborsPtr->begin(); neighborIter != neighborsPtr->end(); ++neighborIter) {
    int32_t neighborTag = *neighborIter;
    
    if (lockedTablePtr && (lockedTablePtr->count(neighborTag) != 0)) {
      // If a locked down table is provided then do not consider a neighbor that appears
      // in the locked table.
      
      if (debug) {
        cout << "skipping consideration of locked neighbor " << neighborTag << endl;
      }
      
      continue;
    }

    Superpixel *neighborSpPtr = getSuperpixelPtr(neighborTag);
    assert(neighborSpPtr);
    
    if (debug) {
      cout << "compare edge between " << tag << " and " << neighborTag << endl;
    }
    
    // Get edge coordinates that are shared between src and neighbor
    
    vector<pair<int32_t,int32_t> > edgeCoords1;
    vector<pair<int32_t,int32_t> > edgeCoords2;
    
    Superpixel::filterEdgeCoords(srcSpPtr, edgeCoords1, neighborSpPtr, edgeCoords2);
    
    // Gather pixels based on the edge coords only
    
    Mat srcEdgeMat;
    
    Superpixel::fillMatrixFromCoords(inputImg, edgeCoords1, srcEdgeMat);
    
    // Note that inputImg is assumed to be in BGR colorspace here
    
    cvtColor(srcEdgeMat, srcEdgeMat, CV_BGR2Lab);

    Mat neighborEdgeMat;
    
    Superpixel::fillMatrixFromCoords(inputImg, edgeCoords2, neighborEdgeMat);
    
    cvtColor(neighborEdgeMat, neighborEdgeMat, CV_BGR2Lab);

    if (debugDumpSuperpixelEdges) {
      std::ostringstream stringStream;
      stringStream << "edge_between_" << tag << "_" << neighborTag << ".png";
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      Mat outputMat = inputImg.clone();
      
      outputMat = Scalar(255, 0, 0);
      
      Mat srcEdgeRed = srcEdgeMat.clone();
      srcEdgeRed = Scalar(0, 0, 255);
      
      Mat neighborEdgeGreen = neighborEdgeMat.clone();
      neighborEdgeGreen = Scalar(0, 255, 0);
      
      Superpixel::reverseFillMatrixFromCoords(srcEdgeRed, false, edgeCoords1, outputMat);
      Superpixel::reverseFillMatrixFromCoords(neighborEdgeGreen, false, edgeCoords2, outputMat);
      
      cout << "write " << filename << " ( " << outputMat.cols << " x " << outputMat.rows << " )" << endl;
      imwrite(filename, outputMat);
    }
    
    // Determine smaller num coords of the two and use that as the N
    
    int numCoordsToCompare = mini((int) edgeCoords1.size(), (int) edgeCoords2.size());

    if (debug) {
      cout << "will compare " << numCoordsToCompare << " coords on edge" << endl;
    }
    
    assert(numCoordsToCompare >= 1);
    
    // One each iteration, select the closest coord and mark that slot as used.
    
    uint8_t neighborEdgeMatUsed[numCoordsToCompare];
    
    for (int j = 0; j < numCoordsToCompare; j++) {
      neighborEdgeMatUsed[j] = false;
    }
    
    double distSum = 0.0;
    int numSum = 0;
    
    for (int i = 0; i < numCoordsToCompare; i++) {
      pair<int32_t, int32_t> srcCoord = edgeCoords1[i];
      Vec3b srcVec = srcEdgeMat.at<Vec3b>(0, i);
      
      // Determine which is the dst coordinates is the closest to this src coord via a distance measure.
      // This should give slightly better results.
      
      double minCoordDist = (double) 0xFFFFFFFF;
      int minCoordOffset = -1;
      
      for (int j = 0; j < numCoordsToCompare; j++) {
        if (neighborEdgeMatUsed[j]) {
          // Already compared to this coord
          continue;
        }
        
        pair<int32_t, int32_t> neighborCoord = edgeCoords2[j];
        
        double coordDist = hypot( neighborCoord.first - srcCoord.first, neighborCoord.second - srcCoord.second );
        
        if (debug) {
          char buffer[1024];
          snprintf(buffer, sizeof(buffer), "coord dist from (%5d, %5d) to (%5d, %5d) is %12.4f",
                   srcCoord.first, srcCoord.second,
                   neighborCoord.first, neighborCoord.second,
                   coordDist);
          cout << (char*)buffer << endl;
        }
        
        if (coordDist < minCoordDist) {
          minCoordDist = coordDist;
          minCoordOffset = j;
        }
      }
      assert(minCoordOffset != -1);
      
      if (debug) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "closest to (%5d, %5d) found as (%5d, %5d) dist is %0.4f",
                 srcCoord.first, srcCoord.second,
                 edgeCoords2[minCoordOffset].first, edgeCoords2[minCoordOffset].second, minCoordDist);
        cout << (char*)buffer << endl;
      }
      
      if (debugDumpSuperpixelEdges) {
        std::ostringstream stringStream;
        stringStream << "edge_between_" << tag << "_" << neighborTag << "_step" << i << ".png";
        std::string str = stringStream.str();
        const char *filename = str.c_str();
        
        Mat outputMat = inputImg.clone();
        
        outputMat = Scalar(255, 0, 0);
        
        Mat srcEdgeRed = srcEdgeMat.clone();
        srcEdgeRed = Scalar(0, 0, 255);
        
        Mat neighborEdgeGreen = neighborEdgeMat.clone();
        neighborEdgeGreen = Scalar(0, 255, 0);
        
        srcEdgeRed.at<Vec3b>(0, i) = Vec3b(255, 255, 255);
        neighborEdgeGreen.at<Vec3b>(0, minCoordOffset) = Vec3b(128, 128, 128);
        
        Superpixel::reverseFillMatrixFromCoords(srcEdgeRed, false, edgeCoords1, outputMat);
        Superpixel::reverseFillMatrixFromCoords(neighborEdgeGreen, false, edgeCoords2, outputMat);
        
        cout << "write " << filename << " ( " << outputMat.cols << " x " << outputMat.rows << " )" << endl;
        imwrite(filename, outputMat);
      }
      
      if (minCoordDist > 1.5) {
        // Not close enough to an available pixel to compare, just skip this src pixel
        // and use the next one without adding to the sum.
        continue;
      }
      
      Vec3b dstVec = neighborEdgeMat.at<Vec3b>(0, minCoordOffset);
      neighborEdgeMatUsed[minCoordOffset] = true;
      
      // Calc color Delta-E distance in 3D vector space
      
      double distance = delta_e_1976(srcVec[0], srcVec[1], srcVec[2],
                                     dstVec[0], dstVec[1], dstVec[2]);
      
      if (debug) {
        int32_t srcPixel = Vec3BToUID(srcVec);
        int32_t dstPixel = Vec3BToUID(dstVec);
        
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "LAB dist between pixels 0x%08X and 0x%08X = %0.12f", srcPixel, dstPixel, distance);
        cout << (char*)buffer << endl;
      }
      
      distSum += distance;
      numSum += 1;
    }
    
    assert(numSum > 0);
    
    double distAve = distSum / numSum;
    
    if (debug) {
      char buffer[1024];
      snprintf(buffer, sizeof(buffer), "Ave LAB dist %0.12f calculated from %0.12f / %d", distAve, distSum, numSum);
      cout << (char*)buffer << endl;
    }
    
    // tuple : DIST NUM_COORDS TAG
    
    CompareNeighborTuple tuple = make_tuple(distAve, neighborSpPtr->coords.size(), neighborTag);
    
    results.push_back(tuple);
  }
  
  // Normalize DIST
  
  if (normalize) {
    double maxDist = 0.0;
    
    for (vector<CompareNeighborTuple>::iterator it = results.begin(); it != results.end(); ++it) {
      CompareNeighborTuple tuple = *it;
      
      double dist = get<0>(tuple);
      
      if (dist > maxDist) {
        maxDist = dist;
      }
    }
    
    for (int i=0; i < results.size(); i++) {
      CompareNeighborTuple tuple = results[i];
      
      double normDist;
      
      if (maxDist == 0.0) {
        // Special case of only 1 edge pixel and identical RGB values
        normDist = 1.0;
      } else {
        normDist = get<0>(tuple) / maxDist;
      }
      
      CompareNeighborTuple normTuple(normDist, get<1>(tuple), get<2>(tuple));
      
      results[i] = normTuple;
    }
  }
  
  if (debug) {
    cout << "unsorted tuples from src superpixel " << tag << endl;
    
    for (vector<CompareNeighborTuple>::iterator it = results.begin(); it != results.end(); ++it) {
      CompareNeighborTuple tuple = *it;
      
      char buffer[1024];
      snprintf(buffer, sizeof(buffer), "( %12.4f %5d %5d )", get<0>(tuple), get<1>(tuple), get<2>(tuple));
      cout << (char*)buffer << endl;
    }
  }
  
  // Sort tuples by DIST value with ties in increasing order
  
  if (results.size() > 1) {
    sort(results.begin(), results.end(), CompareNeighborTupleFunc);
  }
  
  if (debug || debugShowSorted) {
    cout << "sorted tuples from src superpixel " << tag << endl;

    for (vector<CompareNeighborTuple>::iterator it = results.begin(); it != results.end(); ++it) {
      CompareNeighborTuple tuple = *it;
      
      char buffer[1024];
      snprintf(buffer, sizeof(buffer), "( %12.4f %5d %5d )", get<0>(tuple), get<1>(tuple), get<2>(tuple));
      cout << (char*)buffer << endl;
    }
  }
  
  return;
}

// This method is invoked to do a histogram based backprojection to return alikeness
// info about the neighbors of the superpixel. This method uses a histogram based compare
// to return a list sorted by decreasing normalized value determined by averaging the
// backprojection percentages for each pixel.
//
// inputImg : image pixels to read from (assumed to be BGR)
// tag      : superpixel tag that neighbors will be looked up from
// results  : list of neighbors that fit the accept percentage
// locked   : table of superpixel tags already locked
// step     : count of number of steps used for debug output
// conversion: colorspace to convert to before hist calculations
// numPercentRanges : N to indicate the uniform breakdown of
//          percentage values. For example, if numPercentRanges=20
//          then the fill range of 0.0 -> 1.0 is treated as 20
//          ranges covering 5% prob each.
// numTopPercent: number of percentage slots that are acceptable.
//          For 20 ranges and 2 slots, each slot covers 5% so the
//          total acceptable range is then 10%.
// minGraylevel: A percentage value must be GTEQ this grayscale
//          prob value to be considered.
//
// Return tuples : (PERCENT NUM_COORDS TAG)

void
SuperpixelImage::backprojectNeighborSuperpixels(Mat &inputImg,
                                                int32_t tag,
                                                vector<CompareNeighborTuple> &results,
                                                unordered_map<int32_t, bool> *lockedTablePtr,
                                                int32_t step,
                                                int conversion,
                                                int numPercentRanges,
                                                int numTopPercent,
                                                bool roundPercent,
                                                int minGraylevel,
                                                int numBins)
{
  const bool debug = false;
  const bool debugDumpSuperpixels = false;
  const bool debugShowSorted = false;

  const bool debugDumpAllBackProjection = false;
  
  const bool debugDumpCombinedBackProjection = false;
  
  vector<int32_t> *neighborsPtr = edgeTable.getNeighborsPtr(tag);
  
  assert(lockedTablePtr);
  
  if (!results.empty()) {
    results.erase (results.begin(), results.end());
  }
  
  // Before parsing histogram and emitting intermediate images check for the case where a superpixel
  // has all locked neighbors and return early without doing anything in this case. The locked table
  // check is very fast and the number of histogram parses avoided is large.
  
  bool allNeighborsLocked = true;
  
  for (vector<int32_t>::iterator neighborIter = neighborsPtr->begin(); neighborIter != neighborsPtr->end(); ++neighborIter) {
    int32_t neighborTag = *neighborIter;
    
    if (lockedTablePtr->count(neighborTag) != 0) {
      // Neighbor is locked
    } else {
      // Neighbor is not locked
      allNeighborsLocked = false;
      break;
    }
  }
  
  if (allNeighborsLocked) {
    if (debug) {
      cout << "early return from backprojectNeighborSuperpixels since all neighbors are locked" << endl;
    }
    
    return;
  }
  
  Mat srcSuperpixelMat;
  Mat srcSuperpixelHist;
  Mat srcSuperpixelBackProjection;
  
  // Read RGB pixels for the largest superpixel identified by tag from the input image.
  // Gen histogram and then create a back projected output image that shows the percentage
  // values for each pixel in the connected neighbors.
  
  fillMatrixFromCoords(inputImg, tag, srcSuperpixelMat);
  
  if (debugDumpAllBackProjection == true) {
    // Create histogram and generate back projection for entire image
    parse3DHistogram(&srcSuperpixelMat, &srcSuperpixelHist, &inputImg, &srcSuperpixelBackProjection, conversion, numBins);
  } else {
    // Create histogram but do not generate back projection for entire image
    parse3DHistogram(&srcSuperpixelMat, &srcSuperpixelHist, NULL, NULL, conversion, numBins);
  }
  
  if (debugDumpSuperpixels) {
    std::ostringstream stringStream;
    if (step == -1) {
      stringStream << "superpixel_" << tag << ".png";
    } else {
      stringStream << "superpixel_step_" << step << "_" << tag << ".png";
    }
    std::string str = stringStream.str();
    const char *filename = str.c_str();
    
    cout << "write " << filename << " ( " << srcSuperpixelMat.cols << " x " << srcSuperpixelMat.rows << " )" << endl;
    imwrite(filename, srcSuperpixelMat);
  }
  
  if (debugDumpAllBackProjection) {
    std::ostringstream stringStream;
    if (step == -1) {
      stringStream << "backproject_from" << tag << ".png";
    } else {
      stringStream << "backproject_step_" << step << "_from_" << tag << ".png";
    }
    std::string str = stringStream.str();
    const char *filename = str.c_str();
    
    cout << "write " << filename << " ( " << srcSuperpixelBackProjection.cols << " x " << srcSuperpixelBackProjection.rows << " )" << endl;
    
    imwrite(filename, srcSuperpixelBackProjection);
  }
  
  if (debugDumpCombinedBackProjection) {
    // Use this image to fill in all back projected values for neighbors
    
    Scalar bg = Scalar(255,0,0); // Blue
    
    Mat origSize(inputImg.size(), CV_8UC(3), bg);
    srcSuperpixelBackProjection = origSize;
    
    Mat srcSuperpixelGreen = srcSuperpixelMat;
    srcSuperpixelGreen = Scalar(0,255,0);
    
    reverseFillMatrixFromCoords(srcSuperpixelGreen, false, tag, srcSuperpixelBackProjection);
  }
  
  for (vector<int32_t>::iterator neighborIter = neighborsPtr->begin(); neighborIter != neighborsPtr->end(); ++neighborIter) {
    // Do back projection on neighbor pixels using histogram from biggest superpixel
    
    int32_t neighborTag = *neighborIter;
    
    if (lockedTablePtr && (lockedTablePtr->count(neighborTag) != 0)) {
      // If a locked down table is provided then do not consider a neighbor that appears
      // in the locked table.
      
      if (debug) {
        cout << "skipping consideration of locked neighbor " << neighborTag << endl;
      }
      
      continue;
    }
    
    Mat neighborSuperpixelMat;
    //Mat neighborSuperpixelHist;
    Mat neighborBackProjection;
    
    // Back project using the 3D histogram parsed from the largest superpixel only
    
    fillMatrixFromCoords(inputImg, neighborTag, neighborSuperpixelMat);
      
    parse3DHistogram(NULL, &srcSuperpixelHist, &neighborSuperpixelMat, &neighborBackProjection, conversion, numBins);
    
    if (debugDumpSuperpixels) {
      std::ostringstream stringStream;
      stringStream << "superpixel_" << neighborTag << ".png";
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << neighborSuperpixelMat.cols << " x " << neighborSuperpixelMat.rows << " )" << endl;
      imwrite(filename, neighborSuperpixelMat);
    }
    
    if (debugDumpAllBackProjection) {
      // BackProject prediction for just the pixels in the neighbor as compared to the src. Pass
      // the input image generated from the neighbor superpixel and then recreate the original
      // pixel layout by writing the pixels back to the output image in the same order.
      
      std::ostringstream stringStream;
      if (step == -1) {
        stringStream << "backproject_neighbor_" << neighborTag << "_from" << tag << ".png";
      } else {
        stringStream << "backproject_step_" << step << "_neighbor_" << neighborTag << "_from_" << tag << ".png";
      }
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      // The back projected input is normalized grayscale as a float value
      
      //cout << "neighborBackProjection:" << endl << neighborBackProjection << endl;
      
      Mat neighborBackProjectionGrayOrigSize(inputImg.size(), CV_8UC(3), (Scalar)0);
      
      reverseFillMatrixFromCoords(neighborBackProjection, true, neighborTag, neighborBackProjectionGrayOrigSize);
      
      cout << "write " << filename << " ( " << neighborBackProjectionGrayOrigSize.cols << " x " << neighborBackProjectionGrayOrigSize.rows << " )" << endl;
      
      imwrite(filename, neighborBackProjectionGrayOrigSize);
    }
    
    if (debugDumpCombinedBackProjection) {
      // Write combined back projection values to combined image.

      reverseFillMatrixFromCoords(neighborBackProjection, true, neighborTag, srcSuperpixelBackProjection);
    }
    
    // Threshold the neighbor pixels and then choose a path for expansion that considers all the neighbors
    // via a fill. Any value larger than 200 becomes 255 while any value below becomes zero
    
    /*
    
    threshold(neighborBackProjection, neighborBackProjection, 200.0, 255.0, THRESH_BINARY);
    
    if (debugDumpBackProjection) {
    
      std::ostringstream stringStream;
      if (step == -1) {
        stringStream << "backproject_threshold_neighbor_" << neighborTag << "_from" << tag << ".png";
      } else {
        stringStream << "backproject_threshold_step_" << step << "_neighbor_" << neighborTag << "_from_" << tag << ".png";
      }
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << neighborBackProjection.cols << " x " << neighborBackProjection.rows << " )" << endl;
    
      imwrite(filename, neighborBackProjection);
    }
    
    */
     
    // If more than 95% of the back projection threshold values are on then treat this neighbor superpixel
    // as one that should be merged in this expansion step.
    
    if (1) {
      //const int minGraylevel = 200;
      //const float minPercent = 0.95f;
      
      float oneRange = (1.0f / numPercentRanges);
      float minPercent = 1.0f - (oneRange * numTopPercent);
      
      int count = 0;
      int N = neighborBackProjection.cols;
      
      assert(neighborBackProjection.rows == 1);
      for (int i = 0; i < N; i++) {
        uint8_t gray = neighborBackProjection.at<uchar>(0, i);
        if (gray >= minGraylevel) {
          count += 1;
        }
      }
      
      float per = ((double)count) / N;
      
      if (debug) {
        cout << setprecision(3); // 3.141
        cout << showpoint;
        cout << setw(10);
        
        cout << "for neighbor " << neighborTag << " found " << count << " non-zero out of " << N << " pixels : per " << per << endl;
      }
      
      if (per >= minPercent) {
        if (debug) {
          cout << "added neighbor to merge list" << endl;
        }
        
        // If roundPercent is true then round the percentage in terms of the width of percentage range.
        
        if (roundPercent) {
          float rounded = round(per / oneRange) * oneRange;
          
          if (debug) {
            char buffer[1024];
            snprintf(buffer, sizeof(buffer), "rounded per %0.4f to %0.4f", per, rounded);
            cout << (char*)buffer << endl;
          }
          
          per = rounded;
        }
        
        CompareNeighborTuple tuple(per, N, neighborTag);
        
        results.push_back(tuple);
      }
    }
    
  } // end neighbors loop
  
  if (debug) {
    cout << "unsorted tuples (N = " << results.size() << ") from src superpixel " << tag << endl;
    
    for (vector<CompareNeighborTuple>::iterator it = results.begin(); it != results.end(); ++it) {
      CompareNeighborTuple tuple = *it;
      char buffer[1024];
      snprintf(buffer, sizeof(buffer), "(%12.4f, %5d, %5d)",
               get<0>(tuple), get<1>(tuple), get<2>(tuple));
      cout << (char*)buffer << endl;
    }
  }
  
  // Sort tuples by percent value

  if (results.size() > 1) {
    sort(results.begin(), results.end(), CompareNeighborTupleDecreasingFunc);
  }
  
  if (debug || debugShowSorted) {
    cout << "sorted tuples (N = " << results.size() << ") from src superpixel " << tag << endl;
    
    for (vector<CompareNeighborTuple>::iterator it = results.begin(); it != results.end(); ++it) {
      CompareNeighborTuple tuple = *it;
      char buffer[1024];
      snprintf(buffer, sizeof(buffer), "(%12.4f, %5d, %5d)",
               get<0>(tuple), get<1>(tuple), get<2>(tuple));
      cout << (char*)buffer << endl;
    }
  }
  
  if (debugDumpCombinedBackProjection) {
    std::ostringstream stringStream;
    if (step == -1) {
      stringStream << "backproject_combined_from" << tag << ".png";
    } else {
      stringStream << "backproject_combined_step_" << step << "_from_" << tag << ".png";
    }
    std::string str = stringStream.str();
    const char *filename = str.c_str();
    
    cout << "write " << filename << " ( " << srcSuperpixelBackProjection.cols << " x " << srcSuperpixelBackProjection.rows << " )" << endl;
    
    imwrite(filename, srcSuperpixelBackProjection);
  }
  
  return;
}

// This method will back project from a src superpixel and find all neighbor superpixels that contain
// non-zero back projection values. This back projection is like a flood fill except that it operates
// on histogram percentages. This method returns a list of superpixel ids gathered.

void
SuperpixelImage::backprojectDepthFirstRecurseIntoNeighbors(Mat &inputImg,
                                                           int32_t tag,
                                                           vector<int32_t> &results,
                                                           unordered_map<int32_t, bool> *lockedTablePtr,
                                                           int32_t step,
                                                           int conversion,
                                                           int numPercentRanges,
                                                           int numTopPercent,
                                                           int minGraylevel,
                                                           int numBins)
{
  const bool debug = false;
  const bool debugDumpSuperpixels = false;
  
  const bool debugDumpAllBackProjection = false;
  
  const bool debugDumpCombinedBackProjection = false;
  
  vector<int32_t> *neighborsPtr = edgeTable.getNeighborsPtr(tag);
  
  assert(lockedTablePtr);
  
  if (!results.empty()) {
    results.erase (results.begin(), results.end());
  }
  
  // Before parsing histogram and emitting intermediate images check for the case where a superpixel
  // has all locked neighbors and return early without doing anything in this case. The locked table
  // check is very fast and the number of histogram parses avoided is large.
  
  bool allNeighborsLocked = true;
  
  for (vector<int32_t>::iterator neighborIter = neighborsPtr->begin(); neighborIter != neighborsPtr->end(); ++neighborIter) {
    int32_t neighborTag = *neighborIter;
    
    if (lockedTablePtr->count(neighborTag) != 0) {
      // Neighbor is locked
    } else {
      // Neighbor is not locked
      allNeighborsLocked = false;
      break;
    }
  }
  
  if (allNeighborsLocked) {
    if (debug) {
      cout << "early return from backprojectDepthFirstRecurseIntoNeighbors since all neighbors are locked" << endl;
    }
    
    return;
  }
  
  Mat srcSuperpixelMat;
  Mat srcSuperpixelHist;
  Mat srcSuperpixelBackProjection;
  
  // Read RGB pixels for the largest superpixel identified by tag from the input image.
  // Gen histogram and then create a back projected output image that shows the percentage
  // values for each pixel in the connected neighbors.
  
  fillMatrixFromCoords(inputImg, tag, srcSuperpixelMat);
  
  if (debugDumpAllBackProjection == true) {
    // Create histogram and generate back projection for entire image
    parse3DHistogram(&srcSuperpixelMat, &srcSuperpixelHist, &inputImg, &srcSuperpixelBackProjection, conversion, numBins);
  } else {
    // Create histogram but do not generate back projection for entire image
    parse3DHistogram(&srcSuperpixelMat, &srcSuperpixelHist, NULL, NULL, conversion, numBins);
  }
  
  if (debugDumpSuperpixels) {
    std::ostringstream stringStream;
    if (step == -1) {
      stringStream << "superpixel_" << tag << ".png";
    } else {
      stringStream << "superpixel_step_" << step << "_" << tag << ".png";
    }
    std::string str = stringStream.str();
    const char *filename = str.c_str();
    
    cout << "write " << filename << " ( " << srcSuperpixelMat.cols << " x " << srcSuperpixelMat.rows << " )" << endl;
    imwrite(filename, srcSuperpixelMat);
  }
  
  if (debugDumpAllBackProjection) {
    std::ostringstream stringStream;
    if (step == -1) {
      stringStream << "backproject_from" << tag << ".png";
    } else {
      stringStream << "backproject_step_" << step << "_from_" << tag << ".png";
    }
    std::string str = stringStream.str();
    const char *filename = str.c_str();
    
    cout << "write " << filename << " ( " << srcSuperpixelBackProjection.cols << " x " << srcSuperpixelBackProjection.rows << " )" << endl;
    
    imwrite(filename, srcSuperpixelBackProjection);
  }
  
  if (debugDumpCombinedBackProjection) {
    // Use this image to fill in all back projected values for neighbors
    
    Scalar bg = Scalar(255,0,0); // Blue
    
    Mat origSize(inputImg.size(), CV_8UC(3), bg);
    srcSuperpixelBackProjection = origSize;
    
    Mat srcSuperpixelGreen = srcSuperpixelMat;
    srcSuperpixelGreen = Scalar(0,255,0);
    
    reverseFillMatrixFromCoords(srcSuperpixelGreen, false, tag, srcSuperpixelBackProjection);
  }

  // Table of superpixels already seen via DFS as compared to src superpixel.
  
  unordered_map<int32_t, bool> seenTable;
  
  seenTable[tag] = true;
  
  // Fill queue with initial neighbors of this superpixel
  
  vector<int32_t> queue;
  
  for (vector<int32_t>::iterator neighborIter = neighborsPtr->begin(); neighborIter != neighborsPtr->end(); ++neighborIter) {
    int32_t neighborTag = *neighborIter;
    queue.push_back(neighborTag);
    seenTable[neighborTag] = true;
  }
  
  neighborsPtr = NULL;
  
  // This foreach logic must descend into neighbors and then neighbors of neighbors until the backprojection returns
  // zero for all pixels.
  
  for (; 1 ;) {
    if (debug) {
      cout << "pop off front of queue with " << queue.size() << " elements" << endl;
    }
    
    int sizeNow = (int) queue.size();
    if (sizeNow == 0) {
      if (debug) {
        cout << "queue empty, done DFS iteration" << endl;
      }

      break;
    }
    
    if (debug && 0) {
      cout << "queue:" << endl;
      
      for (vector<int32_t>::iterator it = queue.begin(); it != queue.end(); ++it) {
        int32_t neighborTag = *it;
        cout << neighborTag << endl;
      }
    }
    
    // Pop first element off queue
    
    int32_t neighborTag = queue[sizeNow-1];
    queue.erase(queue.end()-1);
    
#if defined(DEBUG)
    int sizeAfterPop = (int) queue.size();
    assert(sizeNow == sizeAfterPop+1 );
#endif // DEBUG

    if (debug) {
      cout << "popped neighbor tag " << neighborTag << endl;
    }
    
    if (lockedTablePtr->count(neighborTag) != 0) {
      // If a locked down table is provided then do not consider a neighbor that appears
      // in the locked table.
      
      if (debug) {
        cout << "skipping consideration of locked neighbor " << neighborTag << endl;
      }
      
      continue;
    }
    
    Mat neighborSuperpixelMat;
    //Mat neighborSuperpixelHist;
    Mat neighborBackProjection;
    
    // Back project using the 3D histogram parsed from the largest superpixel only
    
    fillMatrixFromCoords(inputImg, neighborTag, neighborSuperpixelMat);
    
    parse3DHistogram(NULL, &srcSuperpixelHist, &neighborSuperpixelMat, &neighborBackProjection, conversion, numBins);
    
    if (debugDumpSuperpixels) {
      std::ostringstream stringStream;
      stringStream << "superpixel_" << neighborTag << ".png";
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << neighborSuperpixelMat.cols << " x " << neighborSuperpixelMat.rows << " )" << endl;
      imwrite(filename, neighborSuperpixelMat);
    }
    
    if (debugDumpAllBackProjection) {
      // BackProject prediction for just the pixels in the neighbor as compared to the src. Pass
      // the input image generated from the neighbor superpixel and then recreate the original
      // pixel layout by writing the pixels back to the output image in the same order.
      
      std::ostringstream stringStream;
      if (step == -1) {
        stringStream << "backproject_neighbor_" << neighborTag << "_from" << tag << ".png";
      } else {
        stringStream << "backproject_step_" << step << "_neighbor_" << neighborTag << "_from_" << tag << ".png";
      }
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      // The back projected input is normalized grayscale as a float value
      
      //cout << "neighborBackProjection:" << endl << neighborBackProjection << endl;
      
      Mat neighborBackProjectionGrayOrigSize(inputImg.size(), CV_8UC(3), (Scalar)0);
      
      reverseFillMatrixFromCoords(neighborBackProjection, true, neighborTag, neighborBackProjectionGrayOrigSize);
      
      cout << "write " << filename << " ( " << neighborBackProjectionGrayOrigSize.cols << " x " << neighborBackProjectionGrayOrigSize.rows << " )" << endl;
      
      imwrite(filename, neighborBackProjectionGrayOrigSize);
    }
    
    if (debugDumpCombinedBackProjection) {
      // Write combined back projection values to combined image.
      
      reverseFillMatrixFromCoords(neighborBackProjection, true, neighborTag, srcSuperpixelBackProjection);
    }
        
    // If more than 95% of the back projection threshold values are on then treat this neighbor superpixel
    // as one that should be merged in this expansion step.
    
    if (1) {
      float oneRange = (1.0f / numPercentRanges);
      float minPercent = 1.0f - (oneRange * numTopPercent);
      
      int count = 0;
      int N = neighborBackProjection.cols;
      
      assert(neighborBackProjection.rows == 1);
      for (int i = 0; i < N; i++) {
        uint8_t gray = neighborBackProjection.at<uchar>(0, i);
        if (gray > minGraylevel) {
          count += 1;
        }
      }
      
      float per = ((double)count) / N;
      
      if (debug) {
        cout << setprecision(3); // 3.141
        cout << showpoint;
        cout << setw(10);
        
        cout << "for neighbor " << neighborTag << " found " << count << " above min graylevel out of " << N << " pixels : per " << per << endl;
      }
      
      if (per > minPercent) {
        if (debug) {
          cout << "added neighbor to merge list" << endl;
        }
        
        results.push_back(neighborTag);
        
        // Iterate over all neighbors of this neighbor and insert at the front of the queue
        
        neighborsPtr = edgeTable.getNeighborsPtr(neighborTag);
        
        if (debug) {
          cout << "cheking " << neighborsPtr->size()  << " possible neighbors for addition to DFS queue" << endl;
        }
        
        for (vector<int32_t>::iterator neighborIter = neighborsPtr->begin(); neighborIter != neighborsPtr->end(); ++neighborIter) {
          int32_t neighborTag = *neighborIter;
          
          if (seenTable.count(neighborTag) == 0) {
            seenTable[neighborTag] = true;
            
            if (debug) {
              for (vector<int32_t>::iterator it = queue.begin(); it != queue.end(); ++it) {
                int32_t existingTag = *it;
                if (existingTag == neighborTag) {
                  assert(0);
                }
              }
            }
            
            queue.push_back(neighborTag);
            
            if (debug) {
              cout << "added unseen neighbor " << neighborTag << endl;
            }
          }
        }
      }
      
      // If this neighbor passed the threshold test then emit an image that shows the
      // backprojected prop grayscale over a blue background so that black can still
      // be seen.
      
      if (debugDumpCombinedBackProjection /*&& (per > minPercent)*/) {
        Scalar bg = Scalar(255,0,0); // Blue
        
        Mat dfsBack(inputImg.size(), CV_8UC(3), bg);
        
        Mat srcSuperpixelGreen = srcSuperpixelMat;
        srcSuperpixelGreen = Scalar(0,255,0);
        
        reverseFillMatrixFromCoords(srcSuperpixelGreen, false, tag, dfsBack);
        reverseFillMatrixFromCoords(neighborBackProjection, true, neighborTag, dfsBack);
        
        std::ostringstream stringStream;
        if (step == -1) {
          stringStream << "backproject_dfs_thresh_neighbor_" << neighborTag << "_from_" << tag << ".png";
        } else {
          stringStream << "backproject_dfs_thresh_combined_step_" << step << "_neighbor_" << neighborTag << "_from_" << tag << ".png";
        }
        std::string str = stringStream.str();
        const char *filename = str.c_str();
        
        cout << "write " << filename << " ( " << dfsBack.cols << " x " << dfsBack.rows << " )" << endl;
        
        imwrite(filename, dfsBack);
      }
      
    }
    
  } // end queue not empty loop
  
  if (debugDumpCombinedBackProjection) {
    std::ostringstream stringStream;
    if (step == -1) {
      stringStream << "backproject_combined_from" << tag << ".png";
    } else {
      stringStream << "backproject_combined_step_" << step << "_from_" << tag << ".png";
    }
    std::string str = stringStream.str();
    const char *filename = str.c_str();
    
    cout << "write " << filename << " ( " << srcSuperpixelBackProjection.cols << " x " << srcSuperpixelBackProjection.rows << " )" << endl;
    
    imwrite(filename, srcSuperpixelBackProjection);
  }
  
  if (debugDumpCombinedBackProjection) {
    // Emit an image that shows the src superpixel as green, all from DFS as red, and unvisited as blue
    
    Scalar bg = Scalar(255,0,0); // Blue
    
    Mat dfsScope(inputImg.size(), CV_8UC(3), bg);
    
    Mat srcSuperpixelGreen = srcSuperpixelMat;
    srcSuperpixelGreen = Scalar(0,255,0);
    
    reverseFillMatrixFromCoords(srcSuperpixelGreen, false, tag, dfsScope);
    
    // Iterate over each indicated neighbor to indicate scope

    for (vector<int32_t>::iterator it = results.begin(); it != results.end(); ++it) {
      int32_t resultTag = *it;
      
      Mat resultsSuperpixelMat;
      
      fillMatrixFromCoords(inputImg, resultTag, resultsSuperpixelMat);
      
      // Fill with Red
      resultsSuperpixelMat = Scalar(0,0,255);
      
      reverseFillMatrixFromCoords(resultsSuperpixelMat, false, resultTag, dfsScope);
    }
    
    std::ostringstream stringStream;
    if (step == -1) {
      stringStream << "backproject_dfs_scope_from" << tag << ".png";
    } else {
      stringStream << "backproject_dfs_scope_combined_step_" << step << "_from_" << tag << ".png";
    }
    std::string str = stringStream.str();
    const char *filename = str.c_str();
    
    cout << "write " << filename << " ( " << srcSuperpixelBackProjection.cols << " x " << srcSuperpixelBackProjection.rows << " )" << endl;
    
    imwrite(filename, dfsScope);
    
  }
  
  return;
}

// Scan superpixels looking for the case where all pixels in one superpixel exactly match all
// the superpixels in a neighbor superpixel. This exact matching situation can happen in flat
// image areas so removing the duplication can significantly simplify the graph before the
// more compotationally expensive histogram comparison logic is run on all edges. Reducing the
// number of edges via this more simplified method executes on N superpixels and then
// comparison to neighbors need only be done when the exact same pixels condition is found.

void SuperpixelImage::mergeIdenticalSuperpixels(Mat &inputImg) {
  const bool debug = false;
  
  // Scan list of superpixels and extract a list of the superpixels where all the
  // pixels have the exact same value. Doing this initial scan means that we
  // create a new list that will not be mutated in the case of a superpixel
  // merge.
  
  vector<int32_t> identicalSuperpixels;
  
  for (vector<int32_t>::iterator it = superpixels.begin(); it != superpixels.end(); ++it) {
    int32_t tag = *it;
    
    if (isAllSamePixels(inputImg, tag)) {
      identicalSuperpixels.push_back(tag);
    }
  }
  
  if (debug) {
    cout << "found " << identicalSuperpixels.size() << " superpixels with all identical pixel values" << endl;
  }
  
  for (vector<int32_t>::iterator it = identicalSuperpixels.begin(); it != identicalSuperpixels.end(); ++it ) {
    int32_t tag = *it;
    
    if (getSuperpixelPtr(tag) == NULL) {
      // Check for the edge case of this superpixel being merged into a neighbor as a result
      // of a previous iteration.
      
      if (debug) {
        cout << "identical superpixel " << tag << " was merged away already" << endl;
      }
      
      continue;
    }
    
    // Iterate over all neighbor superpixels and verify that all those pixels match
    // the first pixel value from the known identical superpixel. This loop invokes
    // merge during the loop so the list of neighbors needs to be a copy of the
    // neighbors list since the neighbors list can be changed by the merge.
    
    vector<int32_t> neighbors = edgeTable.getNeighbors(tag);
    
    if (debug) {
      cout << "found neighbors of known identical superpixel " << tag << endl;
      
      for (vector<int32_t>::iterator neighborIter = neighbors.begin(); neighborIter != neighbors.end(); ++neighborIter) {
        int32_t neighborTag = *neighborIter;
        cout << "neighbor " << neighborTag << endl;
      }
    }
    
    // Do a single table lookup for the src superpixel before iterating over neighbors.
    
    Superpixel *spPtr = getSuperpixelPtr(tag);
    assert(spPtr);
    
    for (vector<int32_t>::iterator neighborIter = neighbors.begin(); neighborIter != neighbors.end(); ++neighborIter) {
      int32_t neighborTag = *neighborIter;
    
      if (isAllSamePixels(inputImg, spPtr, neighborTag)) {
        if (debug) {
          cout << "found identical superpixels " << tag << " and " << neighborTag << endl;
        }
        
        SuperpixelEdge edge(tag, neighborTag);
        mergeEdge(edge);
        
        if (getSuperpixelPtr(tag) == NULL) {
          // In the case where the identical superpixel was merged into a neighbor then
          // the neighbors have changed and this iteration has to end.
          
          if (debug) {
            cout << "ending neighbors iteration since " << tag << " was merged into identical neighbor" << endl;
          }
          
          break;
        }
      }
    }
  }
  
  return;
}

// Sort superpixels by size and sort ties so that smaller pixel tag values appear
// before larger tag values.

void SuperpixelImage::sortSuperpixelsBySize()
{
  const bool debug = false;
  
  vector<SuperpixelSortStruct> sortedSuperpixels;
  sortedSuperpixels.reserve(superpixels.size());
  
  for (vector<int32_t>::iterator it = superpixels.begin(); it != superpixels.end(); ++it) {
    int32_t tag = *it;
    SuperpixelSortStruct ss;
    ss.spPtr = getSuperpixelPtr(tag);
    sortedSuperpixels.push_back(ss);
  }
  
  sort(sortedSuperpixels.begin(), sortedSuperpixels.end(), CompareSuperpixelSizeDecreasingFunc);
  
  assert(sortedSuperpixels.size() == superpixels.size());
  
  int i = 0;
  for (vector<SuperpixelSortStruct>::iterator it = sortedSuperpixels.begin(); it != sortedSuperpixels.end(); ++it, i++) {
    SuperpixelSortStruct ss = *it;
    int32_t tag = ss.spPtr->tag;
    superpixels[i] = tag;
    
    if (debug) {
      cout << "sorted superpixel " << i << " has tag " << superpixels[i] << " with N = " << ss.spPtr->coords.size() << endl;
    }
  }
  
  return;
}

// Bredth first merge approach where the largest superpixel merges the next N neighbor
// superpixels that are of equal sameness as determined by a backproject fill on the
// immediate neighbors. Note that this method uses a threshold so that only superpixel
// neighbors that are very much alike will be merged. Also, this approach can be run
// on even very small sized superpixels since a merge will only happen when the neighbor
// is very much alike, so a merge of an oversegmented area will only combine the very
// alike portions.

int SuperpixelImage::mergeBackprojectSuperpixels(Mat &inputImg, int colorspace, int startStep, BackprojectRange range)
{
  const bool debug = false;
  const bool dumpEachMergeStepImage = false;
  
  // Each iteration will examine the list of superpixels and pick the biggest one
  // that is not already locked. Then that superpixel will be expanded until
  // an edge is encountered. Note that the superpixels list can be modified
  // by a merge, so iterate to find the largest one and then use that value.
  // The main loop will iterate until all superpixels are locked and then
  // the locks will be cleared and the iteration will start again using the
  // now larger superpixels. This will continue to merge
  
  bool done = false;
  int mergeIter = startStep;
  
  int numLockClear = 0;
  unordered_map<int32_t, bool> mergesSinceLockClear;
  
  unordered_map<int32_t, bool> locked;
  
  // Do initial sort of the superpixels list so that superpixels are ordered by
  // the number of coordinates in the superpixel. While the sort takes time
  // it means that looking for the largest superpixel can be done by simply
  // finding the next superpixel since superpixels will always be processed
  // from largest to smallest.
  
  sortSuperpixelsBySize();
  
  int sortedListOffset = 0;
  int32_t maxTag = -1;
  
  while (!done) {
    // Get the next superpixel, it will be LTEQ the size of the current superpixel since
    // the superpixels list was sorted and then only deletes would happen via a merge.

#if defined(DEBUG)
    if (sortedListOffset > 0 && sortedListOffset != superpixels.size()) {
      int prevTag = superpixels[sortedListOffset-1];
      assert(maxTag == prevTag);
    }
#endif // DEBUG
    
    int maxSuperpixelOffset = (int) superpixels.size();
    
    if (sortedListOffset == maxSuperpixelOffset) {
      // At end of superpixels list
      maxTag = -1;
    } else {
      // Find next unlocked superpixel

      maxTag = -1;
      while (sortedListOffset < maxSuperpixelOffset) {
        int32_t nextTag = superpixels[sortedListOffset];
        
        if (debug) {
          Superpixel *spPtr = getSuperpixelPtr(nextTag);
          int numCoords = (int) spPtr->coords.size();
          cout << "next max superpixel " << nextTag << " N = " << numCoords << " at offset " << sortedListOffset << endl;
        }
        
        sortedListOffset++;
        
        if (locked[nextTag]) {
          if (debug) {
            cout << "next max superpixel locked" << endl;
          }
        } else {
          // Not locked, use it now
          maxTag = nextTag;
          break;
        }
      }
      
#if defined(DEBUG)
      if (maxTag != -1) {
        bool isLocked = locked[maxTag];
        assert(isLocked == false);
      }
#endif // DEBUG
    }
    
    if (maxTag == -1) {
      if (debug) {
        cout << "checked superpixels but all were locked" << endl;
      }
      
      if (debug) {
        cout << "found that all superpixels are locked with " << superpixels.size() << " superpixels" << endl;
        cout << "mergesSinceLockClear.size() " << mergesSinceLockClear.size() << " numLockClear " << numLockClear << endl;
      }
      
      if (mergesSinceLockClear.size() == 0) {
        done = true;
        continue;
      }
      
      // Delete lock only for superpixels that were expanded in a previous merge run. This avoids
      // having to recheck all superpixels that did not merge the first time while still checking
      // the superpixels that were expanded and could be ready to merge now.
      
      for (unordered_map<int32_t, bool>::iterator it = mergesSinceLockClear.begin(); it != mergesSinceLockClear.end(); ++it) {
        int32_t merged = it->first;

        if (locked.count(merged) == 0) {
          if (debug) {
            cout << "expanded superpixel has no lock entry to erase (it was merged into another superpixel) " << merged << endl;
          }
        } else {
          if (debug) {
            int sizeBefore = (int) locked.size();
            cout << "erase expanded superpixel lock " << merged << endl;
            locked.erase(merged);
            int sizeAfter = (int) locked.size();
            assert(sizeBefore == sizeAfter+1);
          } else {
            locked.erase(merged);
          }
        }
      }
      
      mergesSinceLockClear.clear();
      sortSuperpixelsBySize();
      sortedListOffset = 0;
      numLockClear++;
      continue;
    }
    
    if (debug) {
      Superpixel *spPtr = getSuperpixelPtr(maxTag);
      int numCoords = (int) spPtr->coords.size();
      cout << "found largest superpixel " << maxTag << " with N=" << numCoords << " pixels" << endl;
    }
    
    // Since this superpixel is the largest one currently, merging with another superpixel will always increase the size
    // of this one. The locked table by id logic depends on being able to track a stable UID applied to one specific
    // superpixel, so this approach of using the largest superpixel means that smaller superpixel will always be merged
    // into the current largest superpixel.
    
    while (true) {
      if (debug) {
        cout << "start iter step " << mergeIter << endl;
      }
      
      // Keep top 95% of sameness compare with gray=200 as min value. So, if > 95% of the pixels are a higher level
      // than 200 the superpixel is returned.
      
      vector<CompareNeighborTuple> resultTuples;
      
      if (range == BACKPROJECT_HIGH_FIVE) {
        backprojectNeighborSuperpixels(inputImg, maxTag, resultTuples, &locked, mergeIter, colorspace, 20, 1, false, 200, 16);
      } else if (range == BACKPROJECT_HIGH_FIVE8) {
        backprojectNeighborSuperpixels(inputImg, maxTag, resultTuples, &locked, mergeIter, colorspace, 20, 2, false, 200, 8);
      } else if (range == BACKPROJECT_HIGH_TEN) {
        backprojectNeighborSuperpixels(inputImg, maxTag, resultTuples, &locked, mergeIter, colorspace, 20, 2, false, 200, 16);
      } else if (range == BACKPROJECT_HIGH_15) {
        backprojectNeighborSuperpixels(inputImg, maxTag, resultTuples, &locked, mergeIter, colorspace, 20, 3, false, 200, 16);
      } else if (range == BACKPROJECT_HIGH_20) {
        backprojectNeighborSuperpixels(inputImg, maxTag, resultTuples, &locked, mergeIter, colorspace, 20, 4, false, 200, 16);
      } else if (range == BACKPROJECT_HIGH_50) {
        backprojectNeighborSuperpixels(inputImg, maxTag, resultTuples, &locked, mergeIter, colorspace, 20, 10, false, 128, 8);
      } else {
        assert(0);
      }
    
      // The back project logic here will return a list of neighbor pixels that are more alike
      // than a threshold and are not already locked. It is possible that all neighbors are
      // locked or are not alike enough and in that case 0 superpixel to merge could be returned.
      
      if (resultTuples.size() == 0) {
        if (debug) {
          cout << "no alike or unlocked neighbors so marking this superpixel as locked also" << endl;
        }
        
        locked[maxTag] = true;
        break;
      }
      
      // Merge each alike neighbor
      
      for (vector<CompareNeighborTuple>::iterator it = resultTuples.begin(); it != resultTuples.end(); ++it) {
        CompareNeighborTuple tuple = *it;
      
        int32_t mergeNeighbor = get<2>(tuple);
        
        SuperpixelEdge edge(maxTag, mergeNeighbor);
        
        if (debug) {
          cout << "will merge edge " << edge << endl;
        }
        
        mergeEdge(edge);
        mergeIter += 1;
        mergesSinceLockClear[maxTag] = true;
        
#if defined(DEBUG)
        // This must never fail since the merge should always consume the other superpixel
        assert(getSuperpixelPtr(maxTag) != NULL);
#endif // DEBUG
        
        if (dumpEachMergeStepImage) {
          Mat resultImg = inputImg.clone();
          resultImg = (Scalar) 0;
          
          writeTagsWithStaticColortable(*this, resultImg);
          
          std::ostringstream stringStream;
          stringStream << "backproject_merge_step_" << mergeIter << ".png";
          std::string str = stringStream.str();
          const char *filename = str.c_str();
          
          imwrite(filename, resultImg);
          
          cout << "wrote " << filename << endl;
        }
      }
      
      if (debug) {
        cout << "done with merge of " << resultTuples.size() << " edges" << endl;
      }
      
    } // end of while true loop
    
  } // end of while (!done) loop
  
  if (debug) {
    cout << "left backproject loop with " << superpixels.size() << " merged superpixels and step " << mergeIter << endl;
  }
  
  return mergeIter;
}

// When doing a BFS expansion of a superpixel each edge must be checked to ensure that
// an edge weight has been computed.

void SuperpixelImage::checkNeighborEdgeWeights(Mat &inputImg,
                                               int32_t tag,
                                               vector<int32_t> *neighborsPtr,
                                               unordered_map<SuperpixelEdge, float> &edgeStrengthMap,
                                               int step)
{
  const bool debug = false;
  
  // If any edges of this superpixel do not have an edge weight then store
  // the neighbor so that it will be considered in a call to compare
  // neighbor edge weights.
  
  if (neighborsPtr == NULL) {
    neighborsPtr = edgeTable.getNeighborsPtr(tag);
  }
  
  bool doNeighborsEdgeCalc = false;
  vector<int32_t> neighborsThatHaveEdgeWeights;
  
  for (vector<int32_t>::iterator neighborIter = neighborsPtr->begin(); neighborIter != neighborsPtr->end(); ++neighborIter) {
    int32_t neighborTag = *neighborIter;
    
#if defined(DEBUG)
    Superpixel *neighborSpPtr = getSuperpixelPtr(neighborTag);
    assert(neighborSpPtr);
#endif // DEBUG
  
    SuperpixelEdge edge(tag, neighborTag);
    
    if (edgeStrengthMap.find(edge) == edgeStrengthMap.end()) {
      // Edge weight does not yet exist for this edge
      
      if (debug) {
        cout << "calculate edge weight for " << edge << endl;
      }
      
      doNeighborsEdgeCalc = true;
    } else {
      if (debug) {
        cout << "edge weight already calculated for " << edge << endl;
      }
      
      neighborsThatHaveEdgeWeights.push_back(neighborTag);
    }
  }
  
  // For each neighbor to calculate, do the computation and then save as edge weight
  
  if (doNeighborsEdgeCalc) {
    vector<CompareNeighborTuple> compareNeighborEdgesVec;

    unordered_map<int32_t, bool> lockedNeighbors;
    
    for (vector<int32_t>::iterator neighborIter = neighborsThatHaveEdgeWeights.begin(); neighborIter != neighborsThatHaveEdgeWeights.end(); ++neighborIter) {
      int32_t neighborTag = *neighborIter;
      lockedNeighbors[neighborTag] = true;
      
      if (debug) {
        cout << "edge weight search locked neighbor " << neighborTag << " since it already has an edge weight" << endl;
      }
    }
    
    unordered_map<int32_t, bool> *lockedPtr = NULL;

    if (lockedNeighbors.size() > 0) {
      lockedPtr = &lockedNeighbors;
    }
    
    compareNeighborEdges(inputImg, tag, compareNeighborEdgesVec, lockedPtr, step, false);
    
    // Create edge weight table entry for each neighbor that was compared
    
    for (vector<CompareNeighborTuple>::iterator it = compareNeighborEdgesVec.begin(); it != compareNeighborEdgesVec.end(); ++it) {
      CompareNeighborTuple tuple = *it;
      float edgeWeight = get<0>(tuple);
      int32_t neighborTag = get<2>(tuple);
      
      SuperpixelEdge edge(tag, neighborTag);
      
      edgeStrengthMap[edge] = edgeWeight;
      
      if (debug) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "for edge %14s calc and saved edge weight %8.4f", edge.toString().c_str(), edgeStrengthMap[edge]);
        cout << (char*)buffer << endl;
      }
    }
  } // end if neighbors block

  return;
}

// Bredth first search to fully expand the largest superpixel in a BFS order
// and then lock the superpixel before expanding in terms of smaller superpixels. This
// logic looks for possible expansion using back projection but it keeps track of
// edge weights so that an edge will not be collapsed when it has a very high weight
// as compared to the other edge weights for this specific superpixel.

int SuperpixelImage::mergeBredthFirst(Mat &inputImg, int colorspace, int startStep, vector<int32_t> *largeSuperpixelsPtr, int numBins)
{
  const bool debug = false;
  const bool dumpLockedSuperpixels = false;
  const bool dumpEachMergeStepImage = false;
  
  vector<int32_t> largeSuperpixels;
  
  if (largeSuperpixelsPtr != NULL) {
    largeSuperpixels = *largeSuperpixelsPtr;
  }
  
  if (debug) {
    cout << "large superpixels before BFS" << endl;
    
    for (vector<int32_t>::iterator it = largeSuperpixels.begin(); it != largeSuperpixels.end(); ++it) {
      int32_t tag = *it;
      cout << tag << endl;
    }
  }
  
  // Each iteration will examine the list of superpixels and pick the biggest one
  // that is not already locked. Then that superpixel will be expanded until
  // an edge is encountered. Note that the superpixels list can be modified
  // by a merge, so iterate to find the largest one and then use that value.
  // The main loop will iterate until all superpixels are locked and then
  // the locks will be cleared and the iteration will start again using the
  // now larger superpixels. This will continue to merge
  
  bool done = false;
  int mergeIter = startStep;
  
  int numLockClear = 0;
  unordered_map<int32_t, bool> mergesSinceLockClear;
  
  unordered_map<int32_t, bool> locked;
  
  // Lock each very large superpixel so that the BFS will expand outward towards the
  // largest superpixels but it will not merge contained superpixels into the existing
  // large ones.
  
  for (vector<int32_t>::iterator it = largeSuperpixels.begin(); it != largeSuperpixels.end(); ++it) {
    int32_t tag = *it;
    locked[tag] = true;
  }
  
  if (dumpLockedSuperpixels) {
    Mat lockedSuperpixelsMask(inputImg.size(), CV_8UC(1), Scalar(0));

    Mat outputTagsImg = inputImg.clone();
    outputTagsImg = (Scalar) 0;
    writeTagsWithStaticColortable(*this, outputTagsImg);
    
    for (vector<int32_t>::iterator it = largeSuperpixels.begin(); it != largeSuperpixels.end(); ++it) {
      int32_t tag = *it;
      
      // Write the largest superpixel tag as the value of the output image.
      
      Mat coordsMat;
      fillMatrixFromCoords(inputImg, tag, coordsMat);
      // Create grayscale version of matrix and set all pixels to 255
      Mat coordsGrayMat(coordsMat.size(), CV_8UC(1));
      coordsGrayMat = Scalar(255);
      reverseFillMatrixFromCoords(coordsGrayMat, true, tag, lockedSuperpixelsMask);
    }
    
    // Use mask to copy colortable image values for just the locked superpixels
    
    //imwrite("tags_colortable_before_mask.png", outputTagsImg);
    //imwrite("locked_mask.png", lockedSuperpixelsMask);
    
    Mat maskedOutput;
    
    outputTagsImg.copyTo(maskedOutput, lockedSuperpixelsMask);
    
    std::ostringstream stringStream;
    stringStream << "tags_locked_before_BFS_" << mergeIter << ".png";
    std::string str = stringStream.str();
    const char *filename = str.c_str();
    
    imwrite(filename, maskedOutput);
    
    cout << "wrote " << filename << endl;
  }
  
  // Do initial sort of the superpixels list so that superpixels are ordered by
  // the number of coordinates in the superpixel. While the sort takes time
  // it means that looking for the largest superpixel can be done by simply
  // finding the next superpixel since superpixels will always be processed
  // from largest to smallest.
  
  sortSuperpixelsBySize();
  
  int sortedListOffset = 0;
  int32_t maxTag = -1;
  
  while (!done) {
    // Get the next superpixel, it will be LTEQ the size of the current superpixel since
    // the superpixels list was sorted and then only deletes would happen via a merge.
    
#if defined(DEBUG)
    if (sortedListOffset > 0 && sortedListOffset != superpixels.size()) {
      int prevTag = superpixels[sortedListOffset-1];
      assert(maxTag == prevTag);
    }
#endif // DEBUG
    
    int maxSuperpixelOffset = (int) superpixels.size();
    
    if (sortedListOffset == maxSuperpixelOffset) {
      // At end of superpixels list
      maxTag = -1;
    } else {
      // Find next unlocked superpixel
      
      maxTag = -1;
      while (sortedListOffset < maxSuperpixelOffset) {
        int32_t nextTag = superpixels[sortedListOffset];
        
        if (debug) {
          Superpixel *spPtr = getSuperpixelPtr(nextTag);
          int numCoords = (int) spPtr->coords.size();
          cout << "next max superpixel " << nextTag << " N = " << numCoords << " at offset " << sortedListOffset << endl;
        }
        
        sortedListOffset++;
        
        if (locked[nextTag]) {
          if (debug) {
            cout << "next max superpixel locked" << endl;
          }
        } else {
          // Not locked, use it now
          maxTag = nextTag;
          break;
        }
      }
      
#if defined(DEBUG)
      if (maxTag != -1) {
        bool isLocked = locked[maxTag];
        assert(isLocked == false);
      }
#endif // DEBUG
    }
    
    if (maxTag == -1) {
      if (debug) {
        cout << "all superpixels were locked" << endl;
      }
      
      if (debug) {
        cout << "found that all superpixels are locked with " << superpixels.size() << " superpixels" << endl;
        cout << "mergesSinceLockClear.size() " << mergesSinceLockClear.size() << " numLockClear " << numLockClear << endl;
      }
      
      if (1) {
        // Do not unlock and then rerun this logic once all superpixels are locked since the BFS
        // will expand a blob out as much as it can be safely expanded.

        if (debug) {
          cout << "skipping unlock and search again when all locked" << endl;
        }
        
        done = true;
        continue;
      }
      
      if (mergesSinceLockClear.size() == 0) {
        done = true;
        continue;
      }
      
      // Delete lock only for superpixels that were expanded in a previous merge run. This avoids
      // having to recheck all superpixels that did not merge the first time while still checking
      // the superpixels that were expanded and could be ready to merge now.
      
      for (unordered_map<int32_t, bool>::iterator it = mergesSinceLockClear.begin(); it != mergesSinceLockClear.end(); ++it) {
        int32_t merged = it->first;
        
        if (locked.count(merged) == 0) {
          if (debug) {
            cout << "expanded superpixel has no lock entry to erase (it was merged into another superpixel) " << merged << endl;
          }
        } else {
          if (debug) {
            int sizeBefore = (int) locked.size();
            cout << "erase expanded superpixel lock " << merged << endl;
            locked.erase(merged);
            int sizeAfter = (int) locked.size();
            assert(sizeBefore == sizeAfter+1);
          } else {
            locked.erase(merged);
          }
        }
      }
      
      mergesSinceLockClear.clear();
      sortSuperpixelsBySize();
      sortedListOffset = 0;
      numLockClear++;
      continue;
    }
    
    if (debug) {
      Superpixel *spPtr = getSuperpixelPtr(maxTag);
      int numCoords = (int) spPtr->coords.size();
      cout << "found largest superpixel " << maxTag << " with N=" << numCoords << " pixels" << endl;
    }
    
    // Since this superpixel is the largest one currently, merging with another superpixel will always increase the size
    // of this one. The locked table by id logic depends on being able to track a stable UID applied to one specific
    // superpixel, so this approach of using the largest superpixel means that smaller superpixel will always be merged
    // into the current largest superpixel.
    
    while (true) {
      if (debug) {
        cout << "start iter step " << mergeIter << " with largest superpixel " << maxTag << endl;
      }
      
      // Gather any neighbors that are at least 50% the same as determined by back projection.
      
      vector<CompareNeighborTuple> resultTuples;
      
      // 20 means slows of 5% percent each, 19 indicates that 95% of values is allowed to match
      
//      backprojectNeighborSuperpixels(inputImg, maxTag, resultTuples, &locked, mergeIter, colorspace, 20, 19, true, 64, numBins);
      
      backprojectNeighborSuperpixels(inputImg, maxTag, resultTuples, &locked, mergeIter, colorspace, 20, 10, true, 128, numBins);
      
      if (debug) {
        cout << "backprojectNeighborSuperpixels() results for src superpixel " << maxTag << endl;
        
        for (vector<CompareNeighborTuple>::iterator it = resultTuples.begin(); it != resultTuples.end(); ++it) {
          CompareNeighborTuple tuple = *it;
          char buffer[1024];
          snprintf(buffer, sizeof(buffer), "(%12.4f, %5d, %5d)",
                   get<0>(tuple), get<1>(tuple), get<2>(tuple));
          cout << (char*)buffer << endl;
        }
      }
      
      // Check for cached neighbor edge weights, this logic must be run each time a neighbor back projection
      // is done since a BFS merge can modify the list of neighbors.
      
      vector<int32_t> *neighborsPtr = edgeTable.getNeighborsPtr(maxTag);
      
      checkNeighborEdgeWeights(inputImg, maxTag, neighborsPtr, edgeTable.edgeStrengthMap, mergeIter);
      
      // The back project logic here will return a list of neighbor pixels that are more alike
      // than a threshold and are not already locked. It is possible that all neighbors are
      // locked or are not alike enough and in that case zero superpixels to merge could be returned.
      
      if (resultTuples.size() == 0) {
        if (debug) {
          cout << "no alike or unlocked neighbors so marking this superpixel as locked also" << endl;
        }
        
        // When a superpixel has no unlocked neighbors check for case where there
        // are no entries at all in the unmerged edge weights list. A superpixel
        // that has no neighbors it could possibly merge with can hit this condition.
        
        Superpixel *spPtr = getSuperpixelPtr(maxTag);
        
        if (spPtr->unmergedEdgeWeights.size() == 0) {
          vector<float> unmergedEdgeWeights;
          
          // Gather cached edge weights, this operation is fast since all the edge weights
          // have been cached already and edge weights are shared between superpixels.
          
          for (vector<int32_t>::iterator neighborIter = neighborsPtr->begin(); neighborIter != neighborsPtr->end(); ++neighborIter) {
            int32_t neighborTag = *neighborIter;
            SuperpixelEdge edge(maxTag, neighborTag);
            float edgeWeight = edgeTable.edgeStrengthMap[edge];
            unmergedEdgeWeights.push_back(edgeWeight);
          }
          
          if (debug) {
            if (unmergedEdgeWeights.size() > 0) {
              cout << "adding unmerged edge weights" << endl;
            }
            
            for (vector<float>::iterator it = unmergedEdgeWeights.begin(); it != unmergedEdgeWeights.end(); ++it) {
              float edgeWeight = *it;              
              char buffer[1024];
              snprintf(buffer, sizeof(buffer), "%12.4f", edgeWeight);
              cout << (char*)buffer << endl;
            }
          }
          
          addUnmergedEdgeWeights(maxTag, unmergedEdgeWeights);
        }
        
        locked[maxTag] = true;
        break;
      }
      
      // Merge each alike neighbor, note that this merge happens in descending probability order
      // and that rounding is used so that bins of 5% each are treated as a group such that larger
      // superpixels in the same percentage bin get merged first.
      
      vector<vector<CompareNeighborTuple>> tuplesSplitIntoBins;
      int totalTuples = 0;
      
      int endIndex = (int) resultTuples.size() - 1;
      
      if (endIndex == 0) {
        // There is only 1 element in resultTuples
        
        vector<CompareNeighborTuple> currentBin;
        
        currentBin.push_back(resultTuples[0]);
        tuplesSplitIntoBins.push_back(currentBin);
        totalTuples += currentBin.size();
      } else {
        vector<CompareNeighborTuple> currentBin;
        
        for ( int i = 0; i < endIndex; i++ ) {
          if (debug && false) {
            cout << "check indexes " << i << " " << (i+1) << endl;
          }
          
          CompareNeighborTuple t0 = resultTuples[i];
          CompareNeighborTuple t1 = resultTuples[i+1];
          
          float currentPer = get<0>(t0);
          float nextPer = get<0>(t1);
          
          if (debug && false) {
            cout << "compare per " << currentPer << " " << nextPer << endl;
          }
          
          if (currentPer == nextPer) {
            currentBin.push_back(t0);
          } else {
            // When different, finish current bin and then clear so
            // that next iteration starts with an empty bin.
            
            currentBin.push_back(t0);
            tuplesSplitIntoBins.push_back(currentBin);
            totalTuples += currentBin.size();
            currentBin.clear();
          }
        }
        
        // Handle the last tuple
        
        CompareNeighborTuple t1 = resultTuples[endIndex];
        currentBin.push_back(t1);
        tuplesSplitIntoBins.push_back(currentBin);
        totalTuples += currentBin.size();
      }
      
      assert(totalTuples == resultTuples.size());
      
      int totalNeighbors = 0;
      if (debug) {
        totalNeighbors = (int) neighborsPtr->size();
      }
      int neighborsMerged = 0;
      
      if (1) {
        // Add unmerged edge weights to stats. Let the stats logic even out small are large edge weights.
        
        unordered_map<int32_t, bool> neighborsThatMightBeMergedTable;
        
        for (vector<CompareNeighborTuple>::iterator it = resultTuples.begin(); it != resultTuples.end(); ++it) {
          CompareNeighborTuple tuple = *it;
          int32_t mergeNeighbor = get<2>(tuple);
          neighborsThatMightBeMergedTable[mergeNeighbor] = true;
        }
        
        vector<float> unmergedEdgeWeights;
        
        // Iterate over each neighbor and lookup the cached edge weight
        
        for (vector<int32_t>::iterator neighborIter = neighborsPtr->begin(); neighborIter != neighborsPtr->end(); ++neighborIter) {
          int32_t neighborTag = *neighborIter;
          SuperpixelEdge edge(maxTag, neighborTag);
#if defined(DEBUG)
          assert(edgeTable.edgeStrengthMap.count(edge) > 0);
#endif // DEBUG
          float edgeWeight = edgeTable.edgeStrengthMap[edge];
          
          if (neighborsThatMightBeMergedTable.count(neighborTag) > 0) {
            // This neighbor might be merged, ignore for now
          } else {
            // This neighbor is known to not be a merge possibility
            
            unmergedEdgeWeights.push_back(edgeWeight);
            
            if (debug) {
              cout << "will add unmergable neighbor edge weight " << edgeWeight << " for neighbor " << neighborTag << endl;
            }
          }
        }
        
        if (unmergedEdgeWeights.size() > 0) {
          addUnmergedEdgeWeights(maxTag, unmergedEdgeWeights);
        }
      }
      
      // Iterate bin by bin. Note that because it is possible that an edge that cannot be merged due to an edge
      // strength being too strong. Continue to process all the bins but just add the edges to the list
      // of unmerged edge weights in that case. As soon as 1 unmerged edge weight is added to unmergedEdgeWeights
      // the the rest of the weights in all the bins are also added.
      
      vector<float> unmergedEdgeWeights;
      
      // Note that the bin by bin merge will merge N neighbors that have the same bin strength but then if there
      // are multiple bins stop so that another round of back projection will be done with the updated histogram
      // results that include the new pixels just merged.
      
      int binOffset = 0;

      for (vector<vector<CompareNeighborTuple>>::iterator binIter = tuplesSplitIntoBins.begin(); binIter != tuplesSplitIntoBins.end(); ++binIter) {
        vector<CompareNeighborTuple> currentBin = *binIter;
        
        if (debug) {
          cout << "will merge per bin" << endl;
          
          for (vector<CompareNeighborTuple>::iterator it = currentBin.begin(); it != currentBin.end(); ++it) {
            CompareNeighborTuple tuple = *it;
            
            char buffer[1024];
            snprintf(buffer, sizeof(buffer), "(%12.4f, %5d, %5d)",
                     get<0>(tuple), get<1>(tuple), get<2>(tuple));
            cout << (char*)buffer << endl;
          }
        }
        
        if (binOffset > 0) {
          // Started processing second bin, exit the neighbor processing loop at this point
          // and do another back projection.
          
          if (debug) {
            cout << "leave bin processing loop in order to do another backprojection" << endl;
          }
          
          break;
        }
        
        binOffset++;
        
        // Edge edge were already computed for all neighbors so gather the
        // edge weights and sort by the edge weight to get the neighbors
        // for just this one bin in edge weight order. This sorting consumes
        // cycles but much of the time there will be few matches per bin so
        // sorting is a no-op.
        
        vector<CompareNeighborTuple> edgeWeightSortedTuples;
        
        for (vector<CompareNeighborTuple>::iterator it = currentBin.begin(); it != currentBin.end(); ++it) {
          CompareNeighborTuple tuple = *it;
          int32_t numCoords = get<1>(tuple);
          int32_t mergeNeighbor = get<2>(tuple);
          SuperpixelEdge edge(maxTag, mergeNeighbor);
#if defined(DEBUG)
          assert(edgeTable.edgeStrengthMap.count(edge) > 0);
#endif // DEBUG
          float edgeWeight = edgeTable.edgeStrengthMap[edge];
          CompareNeighborTuple edgeWeightTuple(edgeWeight, numCoords, mergeNeighbor);
          edgeWeightSortedTuples.push_back(edgeWeightTuple);
        }
        
        if (edgeWeightSortedTuples.size() > 1) {
          sort(edgeWeightSortedTuples.begin(), edgeWeightSortedTuples.end(), CompareNeighborTupleFunc);
        }
        
        if (debug) {
          cout << "edge weight ordered neighbors for this bin" << endl;
          
          for (vector<CompareNeighborTuple>::iterator it = edgeWeightSortedTuples.begin(); it != edgeWeightSortedTuples.end(); ++it) {
            CompareNeighborTuple tuple = *it;
            
            char buffer[1024];
            snprintf(buffer, sizeof(buffer), "(%12.4f, %5d, %5d)",
                     get<0>(tuple), get<1>(tuple), get<2>(tuple));
            cout << (char*)buffer << endl;
          }
        }
        
        for (vector<CompareNeighborTuple>::iterator it = edgeWeightSortedTuples.begin(); it != edgeWeightSortedTuples.end(); ++it) {
          CompareNeighborTuple tuple = *it;
          
          int32_t mergeNeighbor = get<2>(tuple);
          
          SuperpixelEdge edge(maxTag, mergeNeighbor);
          
          // Calc stats for unmerged vs successfully merged edges to determine if this specific edge is
          // a hard edge that should indicate where a large superpixel should stop expanding.
          
          float edgeWeight = get<0>(tuple);
          
          if (unmergedEdgeWeights.size() > 0) {
            if (debug) {
              cout << "continue to merge strong edge for neighbor " << mergeNeighbor << " from bins after strong edge found" << endl;
            }
            
            unmergedEdgeWeights.push_back(edgeWeight);
            continue;
          }
          
          bool shouldMerge = shouldMergeEdge(maxTag, edgeWeight);
          
          if (shouldMerge == false) {
            if (debug) {
              cout << "will not merge edge " << edge << endl;
            }
            
            // Once a strong edge is found that prevents a merge stop iterating over
            // percentage bin values. Be sure to include the rest of the edges in
            // the unmerged stats since later logic needs good stats to know what
            // values represent edges that should not be merged.
            
            if (debug) {
              cout << "superpixel " << maxTag << " found a strong edge, lock superpixel and collect strong edges" << endl;
            }
            
            unmergedEdgeWeights.push_back(edgeWeight);
              
            locked[maxTag] = true;
            
            continue;
          }
          
          if (debug) {
            cout << "will merge edge " << edge << endl;
          }
          
          addMergedEdgeWeight(maxTag, edgeWeight);
          
          mergeEdge(edge);
          mergeIter += 1;
          if (debug) {
          neighborsMerged += 1;
          }
          
          mergesSinceLockClear[maxTag] = true;
          
#if defined(DEBUG)
          // This must never fail since the merge should always consume the other superpixel
          assert(getSuperpixelPtr(maxTag) != NULL);
#endif // DEBUG
          
          if (dumpEachMergeStepImage) {
            Mat resultImg = inputImg.clone();
            resultImg = (Scalar) 0;
            
            writeTagsWithStaticColortable(*this, resultImg);
            
            std::ostringstream stringStream;
            stringStream << "backproject_merge_step_" << mergeIter << ".png";
            std::string str = stringStream.str();
            const char *filename = str.c_str();
            
            imwrite(filename, resultImg);
            
            cout << "wrote " << filename << endl;
          }
        }        
      } // end loop over bins
      
      if (debug) {
        cout << "done merging neighbors of " << maxTag << " : merged " << neighborsMerged << " of " << totalNeighbors << endl;
      }
      
      if (unmergedEdgeWeights.size() > 0) {
        // A series of edges was collected after a strong edge was found and merges stopped
        addUnmergedEdgeWeights(maxTag, unmergedEdgeWeights);
        break;
      }
    } // end of while true loop
    
  } // end of while (!done) loop
  
  if (debug) {
    cout << "left backproject loop with " << superpixels.size() << " merged superpixels and step " << mergeIter << endl;
  }
  
  return mergeIter;
}

bool SuperpixelImage::shouldMergeEdge(int32_t tag, float edgeWeight)
{
  Superpixel *spPtr = getSuperpixelPtr(tag);
  return spPtr->shouldMergeEdge(edgeWeight);
}

// Each edge weight for a neighbor that cannot be merged is added to a list
// specific to this superpixel.

void SuperpixelImage::addUnmergedEdgeWeights(int32_t tag, vector<float> &edgeWeights)
{
  Superpixel *spPtr = getSuperpixelPtr(tag);
  
  for (vector<float>::iterator it = edgeWeights.begin(); it != edgeWeights.end(); ++it) {
    float val = *it;
    spPtr->unmergedEdgeWeights.push_back(val);
  }
  
  return;
}

void SuperpixelImage::addMergedEdgeWeight(int32_t tag, float edgeWeight)
{
  Superpixel *spPtr = getSuperpixelPtr(tag);
  spPtr->mergedEdgeWeights.push_back(edgeWeight);
  return;
}

// Given a superpixel uid scan the neighbors list and generate a stddev to determine if any of the neighbors
// is significantly larger than other neighbors. Return a vector that contains the large neighbors.

void SuperpixelImage::filterOutVeryLargeNeighbors(int32_t tag, vector<int32_t> &largeNeighbors)
{
  const bool debug = false;
  
  if (debug) {
    cout << "filterOutVeryLargeNeighbors for superpixel " << tag << endl;
  }
  
  largeNeighbors.clear();

  vector<int32_t> *neighborsPtr = edgeTable.getNeighborsPtr(tag);

  vector<CompareNeighborTuple> tuples;
  
  for (vector<int32_t>::iterator neighborIter = neighborsPtr->begin(); neighborIter != neighborsPtr->end(); ++neighborIter) {
    int32_t neighborTag = *neighborIter;
    
    Superpixel *spPtr = getSuperpixelPtr(neighborTag);
    assert(spPtr);
    
    int32_t numCoords = (int32_t) spPtr->coords.size();
    
    if (debug) {
      char buffer[1024];
      snprintf(buffer, sizeof(buffer), "neighbor %10d has N = %10d coords", neighborTag, numCoords);
      cout << (char*)buffer << endl;
    }
    
    // Tuple: (UNUSED, UID, SIZE)
    
    CompareNeighborTuple tuple(0.0f, neighborTag, numCoords);
    tuples.push_back(tuple);
  }
  
  if (tuples.size() > 1) {
    sort(tuples.begin(), tuples.end(), CompareNeighborTupleSortByDecreasingLargestNumCoordsFunc);
  }
  
  // Sorted results are now in decreasing num coords order

  if (debug) {
    char buffer[1024];
    
    cout << "sorted tuples:" << endl;
    
    for (vector<CompareNeighborTuple>::iterator it = tuples.begin(); it != tuples.end(); ++it) {
      CompareNeighborTuple tuple = *it;
      snprintf(buffer, sizeof(buffer), "neighbor %10d has N = %10d coords", get<1>(tuple), get<2>(tuple));
      cout << (char*)buffer << endl;
    }
  }

  vector<float> sizesVec;
  
  while (1) {
    // If there is only 1 element left in the tuples at this point, break right away since
    // there is no need to run stddev on one element.
    
    if (tuples.size() == 1) {
      if (debug) {
        cout << "exit stddev loop since only 1 tuple left" << endl;
      }
      
      break;
    }
    
    float mean, stddev;
  
    sizesVec.clear();
    
    for (vector<CompareNeighborTuple>::iterator it = tuples.begin(); it != tuples.end(); ++it) {
      CompareNeighborTuple tuple = *it;
      int numCoords = get<2>(tuple);
      sizesVec.push_back((float)numCoords);
    }
    
    if (debug) {
      char buffer[1024];
      
      cout << "stddev on " << tuples.size() << " tuples:" << endl;
      
      for (vector<CompareNeighborTuple>::iterator it = tuples.begin(); it != tuples.end(); ++it) {
        CompareNeighborTuple tuple = *it;
        snprintf(buffer, sizeof(buffer), "neighbor %10d has N = %10d coords", get<1>(tuple), get<2>(tuple));
        cout << (char*)buffer << endl;
      }
    }
    
    sample_mean(sizesVec, &mean);
    sample_mean_delta_squared_div(sizesVec, mean, &stddev);
  
    int32_t maxSize = sizesVec[0];
    
    // Larger than 1/2 stddev indicates size is larger than 68% of all others
    
    float stddevMin;

    if (stddev < 1.0f) {
      // A very small stddev means values are very close together or there
      // is only 1 value.
      stddevMin = maxSize;
    } else if (stddev < MaxSmallNumPixelsVal) {
      // A very very small stddev is found when all the values are very close
      // together. Set stddevMin to the max so that no neighbor is ignored.
      stddevMin = maxSize;
    } else {
      stddevMin = mean + (stddev * 0.5f);
    }
    
    if (debug) {
      char buffer[1024];
      snprintf(buffer, sizeof(buffer), "mean      %10.2f, stddev %10.2f", mean, stddev);
      cout << (char*)buffer << endl;
      
      snprintf(buffer, sizeof(buffer), "stddevMin %10.2f, max N  %10d", stddevMin, maxSize);
      cout << (char*)buffer << endl;
    }
    
    if (maxSize > stddevMin) {
      // The current largest neighbor size is significantly larger than the others, ignore it by
      // removing the first element from the tuples vector.
      
      CompareNeighborTuple tuple = tuples[0];
      int32_t neighborTag = get<1>(tuple);
      largeNeighbors.push_back(neighborTag);
      
      tuples.erase(tuples.begin());
      
      if (debug) {
        cout << "erased first element in tuples, size now " << tuples.size() << endl;
      }
      
    } else {
      break; // break out of while loop
    }
  }
  
  if (debug) {
    cout << "filterOutVeryLargeNeighbors returning (count " << largeNeighbors.size() << ") for superpixel " << tag << endl;
    
    for (vector<int32_t>::iterator neighborIter = largeNeighbors.begin(); neighborIter != largeNeighbors.end(); ++neighborIter) {
      int32_t neighborTag = *neighborIter;
      cout << neighborTag << endl;
    }
  }
  
  return;
}

// Scan for small superpixels and merge away from largest neighbors.

int SuperpixelImage::mergeSmallSuperpixels(Mat &inputImg, int colorspace, int startStep)
{
  const bool debug = false;
  
  const int maxSmallNum = MaxSmallNumPixelsVal;
  
  int mergeStep = startStep;
  
  vector<int32_t> smallSuperpixels;
  
  // First, scan for very small superpixels and treat them as edges automatically so that
  // edge pixels scanning need not consider these small pixels.
  
  for (vector<int32_t>::iterator it = superpixels.begin(); it != superpixels.end(); ++it) {
    int32_t tag = *it;
    Superpixel *spPtr = getSuperpixelPtr(tag);
    assert(spPtr);
    
    int numCoords = (int) spPtr->coords.size();
    
    if (numCoords < maxSmallNum) {
      // Treat small superpixels as edges
      smallSuperpixels.push_back(tag);
    }
  }
  
  if (debug) {
    cout << "found " << smallSuperpixels.size() << " very small superpixels" << endl;
  }
  
  for (vector<int32_t>::iterator it = smallSuperpixels.begin(); it != smallSuperpixels.end(); ) {
    int32_t tag = *it;
    
    Superpixel *spPtr = NULL;
    
    spPtr = getSuperpixelPtr(tag);
    if (spPtr == NULL) {
      // Check for the edge case of this superpixel being merged into a neighbor as a result
      // of a previous iteration.
      
      if (debug) {
        cout << "small superpixel " << tag << " was merged away already" << endl;
      }
      
      ++it;
      continue;
    }
    
    // If a superpixel was very small but it has been merged such that it is no longer small
    // then do not do another merge.
    
    if (spPtr->coords.size() >= maxSmallNum) {
      if (debug) {
        cout << "small superpixel " << tag << " is no longer small after merges : N = " << spPtr->coords.size() << endl;
      }
      
      ++it;
      continue;
    }
    
    // Filter out very large neighbors and then merge with most alike small neighbor.
    
    vector<int32_t> largeNeighbors;
    filterOutVeryLargeNeighbors(tag, largeNeighbors);
    
    unordered_map<int32_t, bool> locked;
    unordered_map<int32_t, bool> *lockedPtr = NULL;
    
    for (vector<int32_t>::iterator neighborIter = largeNeighbors.begin(); neighborIter != largeNeighbors.end(); ++neighborIter) {
      int32_t neighborTag = *neighborIter;
      locked[neighborTag] = true;
      
      if (debug) {
        cout << "marking significantly larger neighbor " << neighborTag << " as locked to merge away from larger BG" << endl;
      }
    }
    if (largeNeighbors.size() > 0) {
      lockedPtr = &locked;
    }
    
    // FIXME: use compareNeighborEdges here?
    
    vector<CompareNeighborTuple> results;
    compareNeighborSuperpixels(inputImg, tag, results, lockedPtr, mergeStep);
    
    // Get the neighbor with the min hist compare level, note that this
    // sort will not see a significantly larger neighbor if found.
    
    CompareNeighborTuple minTuple = results[0];
    
    int32_t minNeighbor = get<2>(minTuple); // Get NEIGHBOR_TAG
    
    // In case of a tie, choose the smallest of the ties (they are sorted in decreasing N order)
    
    if (results.size() > 1 && get<0>(minTuple) == get<0>(results[1])) {
      float tie = get<0>(minTuple);
      
      minNeighbor = get<2>(results[1]);
      
      if (debug) {
        cout << "choose smaller tie neighbor " << minNeighbor << endl;
      }
      
      for (int i = 2; i < results.size(); i++) {
        CompareNeighborTuple tuple = results[i];
        if (tie == get<0>(tuple)) {
          // Still tie
          minNeighbor = get<2>(tuple);
          
          if (debug) {
            cout << "choose smaller tie neighbor " << minNeighbor << endl;
          }
        } else {
          // Not a tie
          break;
        }
      }
    }
    
    if (debug) {
    cout << "for superpixel " << tag << " min neighbor is " << minNeighbor << endl;
    }
    
    SuperpixelEdge edge(tag, minNeighbor);
    
    mergeEdge(edge);
    
    mergeStep += 1;
    
    spPtr = getSuperpixelPtr(tag);
    
    if ((spPtr != NULL) && (spPtr->coords.size() < maxSmallNum)) {
      // nop to continue to continue combine with the same superpixel tag
      
      if (debug) {
        cout << "small superpixel " << tag << " was merged but it still contains only " << spPtr->coords.size() << " pixels" << endl;
      }
    } else {
      ++it;
    }
  }
  
  return mergeStep;
}

// Scan for "edgy" superpixels, these are identified as having a very high percentage of edge
// pixels as compare to non-edge pixels. These edgy pixels should be merged with other edgy
// superpixels so that edge between smooth regions get merged into one edgy region. This merge
// should not merge with the smooth region neighbors.

int SuperpixelImage::mergeEdgySuperpixels(Mat &inputImg, int colorspace, int startStep, vector<int32_t> *largeSuperpixelsPtr)
{
  const bool debug = false;
  
  const bool debugDumpEdgeGrayValues = false;
  
  const bool debugDumpEdgySuperpixels = false;
  
  const bool dumpEachMergeStepImage = false;
  
  int mergeStep = startStep;
  
  // Lock passed in list of largest superpixels
  
  vector<int32_t> largeSuperpixels;
  if (largeSuperpixelsPtr != NULL) {
    largeSuperpixels = *largeSuperpixelsPtr;
  }
  
  unordered_map<int32_t, bool> largestLocked;
  
  if (largeSuperpixels.size() > 0) {
    // Lock each very large superpixel so that the BFS will expand outward towards the
    // largest superpixels but it will not merge contained superpixels into the existing
    // large ones.
    
    for (vector<int32_t>::iterator it = largeSuperpixels.begin(); it != largeSuperpixels.end(); ++it) {
      int32_t tag = *it;
      largestLocked[tag] = true;
    }
  }
  
  // Scan for superpixels that are mostly edges
  
  vector<int32_t> edgySuperpixels;
  
  Mat edgeGrayValues;
  
  if (debugDumpEdgeGrayValues) {
    edgeGrayValues.create(inputImg.rows, inputImg.cols, CV_8UC(3));
  }

  // Another idea would be to compare the size of the src superpixel to the size of
  // a neighbor superpixel. If the neighbor is a lot smaller than the src then the
  // src is unlikely to be an edge, since typically the edges have neighbors that
  // are larger or at least roughly the same size as the src.
  
  // First, scan each superpixel to discover each edge percentage. This is the
  // NUM_EDGE_PIXELS / NUM_PIXELS so that this normalized value will be 1.0
  // when every pixel is an edge pixel.
  
  for (vector<int32_t>::iterator it = superpixels.begin(); it != superpixels.end(); ++it) {
    int32_t tag = *it;
    Superpixel *spPtr = getSuperpixelPtr(tag);
    assert(spPtr);
    
    if (debugDumpEdgeGrayValues) {
      edgeGrayValues = Scalar(255, 0, 0); // Blue
    }
    
    int numSrcCoords = (int) spPtr->coords.size();
    
    if (largestLocked.count(tag) > 0) {
      if (debug) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "skipping %d since it is a largest locked superpixel with N = %d coords", tag, numSrcCoords);
        cout << (char*)buffer << endl;
      }
      
      continue;
    }
    
    vector<int32_t> *neighborsPtr = edgeTable.getNeighborsPtr(tag);
    
    // FIXME: might be better to remove this contained in one superpixel check.
    
    if (neighborsPtr->size() == 1) {
      // In the edge case where there is only 1 neighbor this means that the
      // superpixel is fully contained in another superpixel. Ignore this
      // kind of superpixel since often the one neighbor will be the largest
      // superpixel and this logic would always merge with it.
      
      if (debug) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "edgedetect skipping %d since only 1 neighbor", tag);
        cout << (char*)buffer << endl;
      }
      
      continue;
    }
    
    // Collect all coordinates identified as edge pixels from all the neighbors.
    
    vector<pair<int32_t,int32_t>> edgeCoordsVec;
    
    if (debug) {
      char buffer[1024];
      snprintf(buffer, sizeof(buffer), "edgedetect %10d has N = %10d coords", tag, numSrcCoords);
      cout << (char*)buffer << endl;
    }
    
    for (vector<int32_t>::iterator neighborIter = neighborsPtr->begin(); neighborIter != neighborsPtr->end(); ++neighborIter) {
      int32_t neighborTag = *neighborIter;
      
      Superpixel *neighborPtr = getSuperpixelPtr(neighborTag);
      assert(neighborPtr);
      
      // Gen edge pixels
      
      vector<pair<int32_t,int32_t> > edgeCoordsSrc;
      vector<pair<int32_t,int32_t> > edgeCoordsDst;
      
      Superpixel::filterEdgeCoords(spPtr, edgeCoordsSrc, neighborPtr, edgeCoordsDst);
      
      for (vector<pair<int32_t,int32_t> >::iterator coordsIter = edgeCoordsSrc.begin(); coordsIter != edgeCoordsSrc.end(); ++coordsIter) {
        pair<int32_t,int32_t> coord = *coordsIter;
        edgeCoordsVec.push_back(coord);
      }
      
      int32_t numNeighborCoords = (int32_t) spPtr->coords.size();
      int32_t numSrcEdgeCoords = (int32_t) edgeCoordsSrc.size();
      
      float per = numSrcEdgeCoords / ((float) numSrcCoords);
      
      if (debug) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "neighbor %10d has N = %10d coords", neighborTag, numNeighborCoords);
        cout << (char*)buffer << endl;
        
        snprintf(buffer, sizeof(buffer), "neighbor shares N = %10d edge coords with src (%8.4f percent)", numSrcEdgeCoords, per);
        cout << (char*)buffer << endl;
      }
      
      if (debugDumpEdgeGrayValues) {
        Mat neighborSuperpixelGray(1, (int)edgeCoordsDst.size(), CV_8UC(3));
        uint8_t gray = int(round(per * 255.0f));
        neighborSuperpixelGray = Scalar(gray, gray, gray);
        Superpixel::reverseFillMatrixFromCoords(neighborSuperpixelGray, false, edgeCoordsDst, edgeGrayValues);
      }
    } // end of neighbors loop
    
    // Dedup list of coords
    
    vector<pair<int32_t, int32_t> >::iterator searchIter;
    sort(edgeCoordsVec.begin(), edgeCoordsVec.end());
    searchIter = unique(edgeCoordsVec.begin(), edgeCoordsVec.end());
    edgeCoordsVec.erase(searchIter, edgeCoordsVec.end());
    
    float per = ((int)edgeCoordsVec.size()) / ((float) numSrcCoords);
    
    if (debugDumpEdgeGrayValues) {
      std::ostringstream stringStream;
      stringStream << "edgedetect_" << tag << ".png";
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      if (0) {
        // Write Green pixels for src superpixel
        Mat srcSuperpixelGreen;
        fillMatrixFromCoords(inputImg, tag, srcSuperpixelGreen);
        srcSuperpixelGreen = Scalar(0,255,0);
        reverseFillMatrixFromCoords(srcSuperpixelGreen, false, tag, edgeGrayValues);
      }

      if (1) {
        if (debug) {
          char buffer[1024];
          snprintf(buffer, sizeof(buffer), "unique edge coords N = %10d / %10d (%8.4f percent)", (int)edgeCoordsVec.size(), numSrcCoords, per);
          cout << (char*)buffer << endl;
        }
        
        Mat superpixelGray(1, (int)edgeCoordsVec.size(), CV_8UC(3));
        uint8_t gray = int(round(per * 255.0f));
        superpixelGray = Scalar(0, gray, 0);
        Superpixel::reverseFillMatrixFromCoords(superpixelGray, false, edgeCoordsVec, edgeGrayValues);
      }
      
      cout << "write " << filename << " ( " << edgeGrayValues.cols << " x " << edgeGrayValues.rows << " )" << endl;
      imwrite(filename, edgeGrayValues);
    }
    
    if (per > 0.90f) {
      edgySuperpixels.push_back(tag);
    }
  }
  
  if (debug) {
    cout << "found " << edgySuperpixels.size() << " edgy superpixel out of " << (int)superpixels.size() << " total superpixels" << endl;
  }
  
  if (debugDumpEdgySuperpixels) {
    for (vector<int32_t>::iterator it = edgySuperpixels.begin(); it != edgySuperpixels.end(); ++it ) {
      int32_t tag = *it;

      Mat edgyMat = inputImg.clone();
      edgyMat = Scalar(255, 0, 0); // Blue
      
      Mat srcSuperpixelGreen;
      fillMatrixFromCoords(inputImg, tag, srcSuperpixelGreen);
      srcSuperpixelGreen = Scalar(0,255,0);
      reverseFillMatrixFromCoords(srcSuperpixelGreen, false, tag, edgyMat);
      
      std::ostringstream stringStream;
      stringStream << "edgy_superpixel_" << tag << ".png";
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << edgyMat.cols << " x " << edgyMat.rows << " )" << endl;
      imwrite(filename, edgyMat);
    }
  }
  
  unordered_map<int32_t, bool> edgySuperpixelsTable;
  for (vector<int32_t>::iterator it = edgySuperpixels.begin(); it != edgySuperpixels.end(); ++it ) {
    int32_t tag = *it;
    edgySuperpixelsTable[tag] = true;
  }
  
  // Iterate over superpixels detected as edgy, since edgy superpixels will only be merged into
  // other edgy superpixels this logic can merge a specific edgy superpixel multiple times.
  // Looping is implemented by removing the first element from the edgySuperpixelsTable
  // until the table is empty.
  
  while (edgySuperpixelsTable.size() > 0) {
    unordered_map<int32_t, bool>::iterator it = edgySuperpixelsTable.begin();
    int32_t tag = it->first;
    
    if (debug) {
      cout << "first edgy table superpixel in table " << tag << endl;
    }
    
#if defined(DEBUG)
    {
      // If an edgy superpixel is in the table it should exist at this point
      Superpixel *spPtr = getSuperpixelPtr(tag);
      assert(spPtr);
    }
#endif // DEBUG
    
    // Filter out any neighbors that are not edgy superpixels. Note that this
    // implicitly ignores the very large superpixels.
    
    unordered_map<int32_t, bool> lockedNeighbors;
    
    vector<int32_t> *neighborsPtr = edgeTable.getNeighborsPtr(tag);
    
    for (vector<int32_t>::iterator neighborIter = neighborsPtr->begin(); neighborIter != neighborsPtr->end(); ++neighborIter) {
      int32_t neighborTag = *neighborIter;
      
      if (edgySuperpixelsTable.count(neighborTag) == 0) {
        // Not an edgy superpixel
        lockedNeighbors[neighborTag] = true;
        
        if (debug) {
          cout << "edge weight search locked neighbor " << neighborTag << " since it is not an edgy superpixel" << endl;
        }
      }
    }
    
    unordered_map<int32_t, bool> *lockedPtr = &lockedNeighbors;
    
    vector<CompareNeighborTuple> results;
    
    compareNeighborEdges(inputImg, tag, results, lockedPtr, mergeStep, false);
    
    if (results.size() == 0) {
      // It is possible that an edgy superpixel has no neighbors that are
      // also edgy superpixels, just ignore this one.
  
      if (debug) {
        cout << "ignored edgy superpixel that has no other edgy superpixel neighbors" << endl;
      }
      
      edgySuperpixelsTable.erase(it);
      continue;
    }
    
    // Iterate from smallest edge weight to largest stopping if that edge weight should
    // not be merged or if the superpixel was merged into the neighbor.
    
    int mergeStepAtResultsStart = mergeStep;
    
    for (vector<CompareNeighborTuple>::iterator tupleIter = results.begin(); tupleIter != results.end(); ++tupleIter) {
      CompareNeighborTuple tuple = *tupleIter;
      
      float edgeWeight = get<0>(tuple);
      int32_t mergeNeighbor = get<2>(tuple);
      
      if (debug) {
        cout << "for superpixel " << tag << " merge neighbor is " << mergeNeighbor << " with edge wieght " << edgeWeight << endl;
      }
      
      // Bomb out if there are no unmerged values to compare an edge weight to. This should not
      // happen unless the recursive BFS did not find neighbor weights for a superpixel.
      
#if defined(DEBUG)
      {
      Superpixel *spPtr = getSuperpixelPtr(tag);
      assert(spPtr->unmergedEdgeWeights.size() > 0);
      }
#endif // DEBUG
      
      // Calc stats for unmerged vs successfully merged edges to determine if this specific edge is
      // a hard edge that should indicate where a large superpixel should stop expanding.
      
      bool shouldMerge = shouldMergeEdge(tag, edgeWeight);

      if (shouldMerge == false) {
        if (debug) {
          cout << "breaking out of merge loop since neighbor superpixel should not be merged" << endl;
        }
        
        break;
      }
      
      SuperpixelEdge edge(tag, mergeNeighbor);
      
      mergeEdge(edge);
      mergeStep += 1;
      
      if (dumpEachMergeStepImage) {
        Mat resultImg = inputImg.clone();
        resultImg = (Scalar) 0;
        
        writeTagsWithStaticColortable(*this, resultImg);
        
        std::ostringstream stringStream;
        stringStream << "merge_step_" << mergeStep << ".png";
        std::string str = stringStream.str();
        const char *filename = str.c_str();
        
        imwrite(filename, resultImg);
        
        cout << "wrote " << filename << endl;
      }
      
      // Determine if this superpixel was just merged away
      
      Superpixel *spPtr = getSuperpixelPtr(tag);
       
      if (spPtr == NULL) {
        if (debug) {
          cout << "breaking out of edge merge loop since superpixel was merged into larger one" << endl;
        }
        
        edgySuperpixelsTable.erase(it);
        break;
      }
      
      // If tag was merged info neighbor then would not get this far, so
      // neighbor must have been merged into tag. Remove neighbor from
      // the table in this case.
      
#if defined(DEBUG)
      {
        Superpixel *spPtr = getSuperpixelPtr(mergeNeighbor);
        assert(spPtr == NULL);
      }
#endif // DEBUG
      
      edgySuperpixelsTable.erase(mergeNeighbor);
    }
    
    // If the edgy superpixel was merged into another superpixel then the table key would
    // have been removed in the loop above. It is also possible that some merges were done
    // and then merges were stopped as a result of a should merge test failing. Check for
    // the case where no merges were done and then remove the key only in that case.
    
    if (mergeStep == mergeStepAtResultsStart) {
      if (debug) {
        cout << "removing edgy superpixel key since no merges were successful" << endl;
      }
      
      edgySuperpixelsTable.erase(it);
    }

  } // end (edgySuperpixelsTable > 0) loop
  
  return mergeStep;
}

// This util method scans the current list of superpixels and returns the largest superpixels
// using a stddev measure. These largest superpixels are highly unlikely to be useful when
// scanning for edges on smaller elements, for example. This method should be run after
// initial joining has identified the largest superpxiels.

void
SuperpixelImage::scanLargestSuperpixels(vector<int32_t> &results)
{
  const bool debug = false;
  
  const int maxSmallNum = MaxSmallNumPixelsVal;
  
  vector<float> superpixelsSizes;
  vector<uint32_t> superpixelsForSizes;
  
  results.clear();
  
  // First, scan for very small superpixels and treat them as edges automatically so that
  // edge pixels scanning need not consider these small pixels.
  
  for (vector<int32_t>::iterator it = superpixels.begin(); it != superpixels.end(); ++it) {
    int32_t tag = *it;
    Superpixel *spPtr = getSuperpixelPtr(tag);
    assert(spPtr);
    
    int numCoords = (int) spPtr->coords.size();
    
    if (numCoords < maxSmallNum) {
      // Ignore really small superpixels in the stats
    } else {
      superpixelsSizes.push_back((float)numCoords);
      superpixelsForSizes.push_back(tag);
    }
  }
  
  if (debug) {
    cout << "found " << superpixelsSizes.size() << " non-small superpixel sizes" << endl;
    
    vector<float> copySizes = superpixelsSizes;
    
    // Sort descending
    
    sort(copySizes.begin(), copySizes.end(), greater<float>());
    
    for (vector<float>::iterator it = copySizes.begin(); it != copySizes.end(); ++it) {
      cout << *it << endl;
    }
  }
  
  float mean, stddev;
  
  sample_mean(superpixelsSizes, &mean);
  sample_mean_delta_squared_div(superpixelsSizes, mean, &stddev);
  
  if (debug) {
    char buffer[1024];
    
    snprintf(buffer, sizeof(buffer), "mean %0.4f stddev %0.4f", mean, stddev);
    cout << (char*)buffer << endl;

    snprintf(buffer, sizeof(buffer), "1 stddev %0.4f", (mean + (stddev * 0.5f * 1.0f)));
    cout << (char*)buffer << endl;
    
    snprintf(buffer, sizeof(buffer), "2 stddev %0.4f", (mean + (stddev * 0.5f * 2.0f)));
    cout << (char*)buffer << endl;

    snprintf(buffer, sizeof(buffer), "3 stddev %0.4f", (mean + (stddev * 0.5f * 3.0f)));
    cout << (char*)buffer << endl;
  }
  
  // If the stddev is not at least 100 then these pixels are very small and it is unlikely
  // than any one would be significantly larger than the others. Simply return an empty
  // list as results in this case.
  
  const float minStddev = 100.0f;
  if (stddev < minStddev) {
    if (debug) {
      cout << "small stddev " << stddev << " found so returning empty list of largest superpixels" << endl;
    }
    
    return;
  }
  
  float upperLimit = mean + (stddev * 0.5f * 3.0f); // Cover 99.7 percent of the values
  
  if (debug) {
    char buffer[1024];
    
    snprintf(buffer, sizeof(buffer), "upperLimit %0.4f", upperLimit);
    cout << (char*)buffer << endl;
  }
  
  int offset = 0;
  for (vector<float>::iterator it = superpixelsSizes.begin(); it != superpixelsSizes.end(); ++it, offset++) {
    float numCoords = *it;
    
    if (numCoords <= upperLimit) {
      // Ignore this element
      
      if (debug) {
        uint32_t tag = superpixelsForSizes[offset];
        cout << "ignore superpixel " << tag << " with N = " << (int)numCoords << endl;
      }
    } else {
      uint32_t tag = superpixelsForSizes[offset];
      
      if (debug) {
        cout << "keep superpixel " << tag << " with N = " << (int)numCoords << endl;
      }
      
      results.push_back(tag);
    }
  }
  
  return;
}

// This method will examine the bounds of the largest superpixels and then use a backprojection
// to recalculate the exact bounds where the larger smooth area runs into edges defined by
// the smaller superpixels. For example, in many images with an identical background the primary
// edge is defined between the background color and the foreground item(s). This method will
// write a new output image

void SuperpixelImage::rescanLargestSuperpixels(Mat &inputImg, Mat &outputImg, vector<int32_t> *largeSuperpixelsPtr)
{
  const bool debug = false;
  const bool debugDumpSuperpixels = false;
  const bool debugDumpBackprojections = false;
  
  vector<int32_t> largeSuperpixels;
  if (largeSuperpixelsPtr != NULL) {
    largeSuperpixels = *largeSuperpixelsPtr;
  } else {
    scanLargestSuperpixels(largeSuperpixels);
  }

  // Gather superpixels that are larger than the upper limit

  outputImg.create(inputImg.size(), CV_8UC(3));
  outputImg = (Scalar)0;

  for (vector<int32_t>::iterator it = largeSuperpixels.begin(); it != largeSuperpixels.end(); ++it) {
    int32_t tag = *it;
    Superpixel *spPtr = getSuperpixelPtr(tag);
    assert(spPtr);
    
    // Do back projection after trimming the large superpixel range. First, simply emit the large
    // superpixel as an image.
    
    Mat srcSuperpixelMat;
    Mat srcSuperpixelHist;
    Mat srcSuperpixelBackProjection;
    
    // Read RGB pixel data from main image into matrix for this one superpixel and then gen histogram.
    
    fillMatrixFromCoords(inputImg, tag, srcSuperpixelMat);
    
    parse3DHistogram(&srcSuperpixelMat, &srcSuperpixelHist, NULL, NULL, 0, -1);
    
    if (debugDumpSuperpixels) {
      std::ostringstream stringStream;
      stringStream << "superpixel_" << tag << ".png";
      
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      Mat revMat = outputImg.clone();
      revMat = (Scalar) 0;
      
      reverseFillMatrixFromCoords(srcSuperpixelMat, false, tag, revMat);
      
      cout << "write " << filename << " ( " << revMat.cols << " x " << revMat.rows << " )" << endl;
      imwrite(filename, revMat);
    }

    // Generate back projection for entire image
    
    parse3DHistogram(NULL, &srcSuperpixelHist, &inputImg, &srcSuperpixelBackProjection, 0, -1);
    
    // srcSuperpixelBackProjection is a grayscale 1 channel image

    if (debugDumpBackprojections) {
      std::ostringstream stringStream;
      stringStream << "backproject_from_" << tag << ".png";
      
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << srcSuperpixelBackProjection.cols << " x " << srcSuperpixelBackProjection.rows << " )" << endl;
      imwrite(filename, srcSuperpixelBackProjection);
    }
    
    // A back projection is more efficient if we actually know a range to indicate how near to the edge the detected edge line is.
    // But, if the amount it is off is unknown then how to determine the erode range?
    
    // Doing a back projection is more effective if the actualy
    
    // Erode the superpixel shape a little bit to draw it back from the likely edge.
    
    Mat erodeBWMat(inputImg.size(), CV_8UC(1), Scalar(0));
    Mat bwPixels(srcSuperpixelMat.size(), CV_8UC(1), Scalar(255));
    
    // FIXME: rework fill to write to the kind of Mat either color or BW
    
    reverseFillMatrixFromCoords(bwPixels, true, tag, erodeBWMat);
    
    if (debugDumpSuperpixels) {
      std::ostringstream stringStream;
      stringStream << "superpixel_" << tag << "_bw.png";
      
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << erodeBWMat.cols << " x " << erodeBWMat.rows << " )" << endl;
      imwrite(filename, erodeBWMat);
    }
    
    // Erode to pull superpixel edges back by a few pixels
    
//    Mat erodeMinBWMat = erodeBWMat.clone();
//    erodeMinBWMat = (Scalar) 0;
//    Mat erodeMaxBWMat = erodeBWMat.clone();
//    erodeMaxBWMat = (Scalar) 0;

    Mat minBWMat, maxBWMat;
    
    //int erosion_type = MORPH_ELLIPSE;
    int erosion_type = MORPH_RECT;
    
    // FIXME: should the erode size depend on the image dimensions?
    
    int erosion_size = 1;
    
//    if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
//    else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
//    else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
    
    Mat element = getStructuringElement( erosion_type,
                                        Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                        Point( erosion_size, erosion_size ) );
    
    // Apply the erosion operation to reduce the white area
    
    erode( erodeBWMat, minBWMat, element );
    
    if (debugDumpSuperpixels) {
      std::ostringstream stringStream;
      stringStream << "superpixel_" << tag << "_bw_erode.png";
      
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << minBWMat.cols << " x " << minBWMat.rows << " )" << endl;
      imwrite(filename, minBWMat);
    }
    
    // Apply a dilate to expand the white area
    
    dilate( erodeBWMat, maxBWMat, element );

    if (debugDumpSuperpixels) {
      std::ostringstream stringStream;
      stringStream << "superpixel_" << tag << "_bw_dilate.png";
      
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << maxBWMat.cols << " x " << maxBWMat.rows << " )" << endl;
      imwrite(filename, maxBWMat);
    }
    
    // Calculate gradient x 2 which is erode and dialate and then intersection.
    // This area is slightly fuzzy to account for merging not getting exactly
    // on the edge.
    
    Mat gradMat;
    
    morphologyEx(erodeBWMat, gradMat, MORPH_GRADIENT, element, Point(-1,-1), 1);

    if (debugDumpSuperpixels) {
      std::ostringstream stringStream;
      stringStream << "superpixel_" << tag << "_bw_gradient.png";
      
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << gradMat.cols << " x " << gradMat.rows << " )" << endl;
      imwrite(filename, gradMat);
    }
    
    // The white pixels indicate where histogram backprojection should be examined, a mask on whole image.
    // A histogram could be computed from the entire background area, or it could be computed from the
    // area around the identified but known to not be the edge.
    
    int numNonZero = countNonZero(gradMat);
    
    Mat backProjectInputFlatMat(1, numNonZero, CV_8UC(3));
    Mat backProjectOutputFlatMat(1, numNonZero, CV_8UC(1));
    
    vector <pair<int32_t, int32_t>> coords;
    
    int offset = 0;
    
    for( int y = 0; y < gradMat.rows; y++ ) {
      for( int x = 0; x < gradMat.cols; x++ ) {
        uint8_t bVal = gradMat.at<uint8_t>(y, x);
        
        if (bVal) {
          Vec3b pixelVec = inputImg.at<Vec3b>(y, x);
          backProjectInputFlatMat.at<Vec3b>(0, offset++) = pixelVec;
        }
      }
    }
    
    assert(offset == numNonZero);
    
    // Generate back projection for just the mask area.
    
    if (debug) {
      backProjectOutputFlatMat = (Scalar) 0;
    }
    
    parse3DHistogram(NULL, &srcSuperpixelHist, &backProjectInputFlatMat, &backProjectOutputFlatMat, 0, -1);
    
    // copy back projection pixels back into full size image, note that any pixels not in the mask
    // identified by gradMat are ignored.
    
    Mat maskedGradientMat = erodeBWMat.clone();
    maskedGradientMat = (Scalar) 0;
    
    offset = 0;
    
    for( int y = 0; y < gradMat.rows; y++ ) {
      for( int x = 0; x < gradMat.cols; x++ ) {
        uint8_t bVal = gradMat.at<uint8_t>(y, x);
        
        if (bVal) {
          uint8_t per = backProjectOutputFlatMat.at<uint8_t>(0, offset++);
          maskedGradientMat.at<uint8_t>(y, x) = per;
        }
      }
    }
    
    assert(offset == numNonZero);
    
    if (debugDumpSuperpixels) {
      std::ostringstream stringStream;
      stringStream << "superpixel_" << tag << "_bw_gradient_backproj.png";
      
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << maskedGradientMat.cols << " x " << maskedGradientMat.rows << " )" << endl;
      imwrite(filename, maskedGradientMat);
    }
    
    // The generated back projection takes the existing edge around the foreground object into
    // account with this approach since the histogram was created from edge defined by the
    // superpixel segmentation. If the goal is to have the edge right on the detected edge
    // then this would seem to be best. If instead the background is defined by creating a
    // histogram from the area pulled away from the edge then the background histogram would
    // not get as close to the foreground object.
    
  }
  
  return;
}

// Depth first "flood fill" like merge where a source superpixel is used to create a histogram that will
// then be used to do a depth first search for like superpixels. This logic depends on having initial
// superpixels that are alike enough that a depth first merge based on the pixels is actually useful
// and that means alike pixels need to have already been merged into larger superpixels.
//
// kernel is an odd kernel size (3,5,7,9,11,13,15,17)
// cradius is a color radius

void SuperpixelImage::applyBilinearFiltering(Mat &inputImg, Mat &outputImg, int kernel, int cradius)
{
  double sigmaSpace = cradius;
  double maxSigmaColor = cradius;
  
  int kernel_dim = kernel;
  
  Size ksize(kernel_dim, kernel_dim);
  
//  Mat bilat;
  
  adaptiveBilateralFilter ( inputImg, outputImg, ksize, sigmaSpace, maxSigmaColor );
  
  // Run edge detection on results of bilaeral filter, if the bilateral filtering keeps
  // edges while reducing valley diffs then the edge should become more pronounced.
  
  Mat bilatGray;
  Mat grad;
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  
  cvtColor( outputImg, bilatGray, CV_BGR2GRAY );
  
  /// Gradient X
  Scharr( bilatGray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  //Sobel( bilatGray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );
  
  /// Gradient Y
  Scharr( bilatGray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  //Sobel( bilatGray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );
  
  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
  
  imwrite("gradient.png", grad);
}

// Return vector of all edges
vector<SuperpixelEdge> SuperpixelImage::getEdges()
{
  return edgeTable.getAllEdges();
}

// Read RGB values from larger input image and create a matrix that is the width
// of the superpixel and contains just the pixels defined by the coordinates
// contained in the superpixel. The caller passes in the tag from the superpixel
// in question in order to find the coords.

void SuperpixelImage::fillMatrixFromCoords(Mat &input, int32_t tag, Mat &output) {
  Superpixel *spPtr = getSuperpixelPtr(tag);
  spPtr->fillMatrixFromCoords(input, tag, output);
}

// This method is the inverse of fillMatrixFromCoords(), it reads pixel values from a matrix
// and writes them back to the corresponding X,Y values location in an image. This method is
// very useful when running an image operation on all the pixels in a superpixel but without
// having to process all the pixels in a bbox area. The dimensions of the input must be
// NUM_COORDS x 1. The caller must init the matrix values and the matrix size. This method
// can be invoked multiple times to write multiple superpixel values to the same output
// image.

void SuperpixelImage::reverseFillMatrixFromCoords(Mat &input, bool isGray, int32_t tag, Mat &output) {
  Superpixel *spPtr = getSuperpixelPtr(tag);
  spPtr->reverseFillMatrixFromCoords(input, isGray, tag, output);
}

// Read RGB values from larger input image based on coords defined for the superpixel
// and return true only if all the pixels have the exact same value.

bool SuperpixelImage::isAllSamePixels(Mat &input, int32_t tag) {
  const bool debug = false;
  
  Superpixel *spPtr = getSuperpixelPtr(tag);
  
  vector<pair<int32_t,int32_t> > &coords = spPtr->coords;
  
  if (debug) {
    int numCoords = (int) coords.size();
    
    cout << "checking for superpixel all same pixels for " << tag << " with coords N=" << numCoords << endl;
  }
  
  pair<int32_t,int32_t> coord = spPtr->coords[0];
  int32_t X = coord.first;
  int32_t Y = coord.second;
  Vec3b pixelVec = input.at<Vec3b>(Y, X);
  uint32_t knownFirstPixel = (uint32_t) (Vec3BToUID(pixelVec) & 0x00FFFFFF);
  
  return isAllSamePixels(input, knownFirstPixel, coords);
}

// When a superpixel is known to have all identical pixel values then only the first
// pixel in that superpixel needs to be compared to all the other pixels in a second
// superpixel. This optimized code accepts a pointer to the first superpixel in
// order to avoid repeated table lookups.

bool SuperpixelImage::isAllSamePixels(Mat &input, Superpixel *spPtr, int32_t otherTag) {
  Superpixel *otherSpPtr = getSuperpixelPtr(otherTag);
  if (otherSpPtr == NULL) {
    // In the case where a neighbor superpixel was already merged then
    // just return false for the all same test.
    return false;
  }

  // Get pixel value from first coord in first superpixel
  
  pair<int32_t,int32_t> coord = spPtr->coords[0];
  int32_t X = coord.first;
  int32_t Y = coord.second;
  Vec3b pixelVec = input.at<Vec3b>(Y, X);
  uint32_t knownFirstPixel = (uint32_t) (Vec3BToUID(pixelVec) & 0x00FFFFFF);
  
  // Compare first value to all other values in the other superpixel.
  // Performance is critically important here so this code takes care
  // to avoid making a copy of the coords vector or the elements inside
  // the vector since there can be many coordinates.
  
  return isAllSamePixels(input, knownFirstPixel, otherSpPtr->coords);
}

// Read RGB values from larger input image based on coords defined for the superpixel
// and return true only if all the pixels have the exact same value. This method
// accepts a knownFirstPixel value which is a pixel value known to be the first
// value for matching purposes. In the case where two superpixels are being compared
// then determine the known pixel from the first group and pass the coords for the
// second superpixel which will then be checked one by one. This optimization is
// critically important when a very large number of oversegmented superpixel were
// parsed from the original image.

bool SuperpixelImage::isAllSamePixels(Mat &input, uint32_t knownFirstPixel, vector<pair<int32_t,int32_t> > &coords) {
  const bool debug = false;
  
  int numCoords = (int) coords.size();
  assert(numCoords > 0);
  
  if (debug) {
    char buffer[6+1];
    snprintf(buffer, 6+1, "%06X", knownFirstPixel);
    
    cout << "checking for all same pixels with coords N=" << numCoords << " and known first pixel " << buffer << endl;
  }
  
  // FIXME: 32BPP support
  
  for (vector<pair<int32_t,int32_t> >::iterator it = coords.begin(); it != coords.end(); ++it) {
    pair<int32_t,int32_t> coord = *it;
    int32_t X = coord.first;
    int32_t Y = coord.second;
    
    Vec3b pixelVec = input.at<Vec3b>(Y, X);
    uint32_t pixel = (uint32_t) (Vec3BToUID(pixelVec) & 0x00FFFFFF);
    
    if (debug) {
      // Print BGRA format
      
      int32_t pixel = Vec3BToUID(pixelVec);
      char buffer[6+1];
      snprintf(buffer, 6+1, "%06X", pixel);
      
      char ibuf[3+1];
      snprintf(ibuf, 3+1, "%03d", (int)distance(coords.begin(), it));
      
      cout << "pixel i = " << ibuf << " 0xFF" << (char*)&buffer[0] << endl;
    }
    
    if (pixel != knownFirstPixel) {
      if (debug) {
        cout << "pixel differs from known first pixel" << endl;
      }
      
      return false;
    }
  }
  
  if (debug) {
    cout << "all pixels the same after processing " << numCoords << " coordinates" << endl;
  }
  
  return true;
}

// Generate a 3D histogram and or a 3D back projection with the configured settings.
// If a histogram or projection is not wanted then pass NULL.

void parse3DHistogram(Mat *histInputPtr,
                      Mat *histPtr,
                      Mat *backProjectInputPtr,
                      Mat *backProjectPtr,
                      int conversion,
                      int numBins)
{
  const bool debug = false;
  
  if (backProjectPtr) {
    assert(backProjectInputPtr);
    assert(histPtr);
  }
      
  // show non-normalized counts for each bin
  const bool debugCounts = false;

  int imgCount = 1;
  
  Mat mask = Mat();
  
  const int channels[] = {0, 1, 2};
  
  int dims = 3;
  
  int binDim = numBins;
  if (binDim < 0) {
    binDim = 16;
  }
  
  int sizes[] = {binDim, binDim, binDim};
  
  // Range indicates the pixel range (0 <= val < 256)
  
  float rRange[] = {0, 256};
  float gRange[] = {0, 256};
  float bRange[] = {0, 256};
  
  const float *ranges[] = {rRange,gRange,bRange};
  
  bool uniform = true; bool accumulate = false;
  
  // Calculate histogram, note that it is possible to pass
  // in a previously calculated histogram and the result
  // is that back projection will be run with the existing
  // histogram when histInputPtr is passed as NULL
  
  if (histPtr != NULL && histInputPtr != NULL) {
    Mat src;
  
    if (conversion == 0) {
      // Use RGB pixels directly
      src = *histInputPtr;
    } else {
      // Convert input pixel to indicated colorspace before parsing histogram
      cvtColor(*histInputPtr, src, conversion);
    }
  
  assert(!src.empty());
  
  CV_Assert(src.type() == CV_8UC3);
  
  const Mat srcArr[] = {src};
  
  calcHist(srcArr, imgCount, channels, mask, *histPtr, dims, sizes, ranges, uniform, accumulate);
  
  Mat &hist = *histPtr;
    
  if (debug || debugCounts) {
    cout << "histogram:" << endl;
  }
  
  if (debug) {
    cout << "type "<< src.type() << "|" << hist.type() << "\n";
    cout << "rows "<< src.rows << "|" << hist.rows << "\n";
    cout << "columns "<< src.cols << "|" << hist.cols << "\n";
    cout << "channels "<< src.channels() << "|" << hist.channels() << "\n";
  }
  
  assert(histPtr->dims == 3); // binDim x binDim x binDim
  
  int numNonZero;
  int totalNumBins;
  
  numNonZero = 0;
  totalNumBins = 0;
  
  float maxValue = 1.0;
  
  for (int i = 0; i < (binDim * binDim * binDim); i++) {
    totalNumBins++;
    float v = hist.at<float>(0,0,i);
    if (debug) {
      cout << "bin[" << i << "] = " << v << endl;
    }
    if (v != 0.0) {
      if (debug || debugCounts) {
        cout << "bin[" << i << "] = " << v << endl;
      }
      numNonZero++;
      
      if (v > maxValue) {
        maxValue = v;
      }
    }
  }
  
  if (debug) {
    
    cout << "total of " << numNonZero << " non-zero values found in histogram" << endl;
    cout << "total num bins " << totalNumBins  << endl;
    cout << "max bin count val " << maxValue << endl;
    
    cout << "will normalize via mult by 1.0 / " << maxValue << endl;
    
  }
  
  assert(numNonZero > 0); // It should not be possible for no bins to have been filled
  
  hist *= (1.0 / maxValue);
  
  numNonZero = 0;
  
  if (debug) {
    
    for (int i = 0; i < (binDim * binDim * binDim); i++) {
      
      float v = hist.at<float>(0,0,i);
      if (v != 0.0) {
        cout << "nbin[" << i << "] = " << v << endl;
        numNonZero++;
      }
    }
    
    cout << "total of " << numNonZero << " normalized non-zero values found in histogram" << endl;
    
  }
    
  } // end histPtr != NULL
  
  if (backProjectPtr != NULL) {
    // Optionally calculate back projection image (grayscale that shows normalized histogram confidence )
  
    assert(histPtr); // Back projection depends on a histogram
    assert(backProjectInputPtr);
    
    Mat backProjectSrcArr[1];
    
    if (conversion == 0) {
      // Use RGB pixels directly
      backProjectSrcArr[0] = *backProjectInputPtr;
    } else {
      // Convert input pixel to LAB colorspace before parsing histogram
      Mat backProjectInputLab;
      cvtColor(*backProjectInputPtr, backProjectInputLab, conversion);
      backProjectSrcArr[0] = backProjectInputLab;
    }
    
    calcBackProject( backProjectSrcArr, imgCount, channels, *histPtr, *backProjectPtr, ranges, 255.0, uniform );
  }
  
  return;
}

// Given a set of weights that could show positive or negative deltas,
// calculate a bound and determine if the currentWeight falls within
// this bound. This method returns true if the expansion of a superpixel
// should continue, false if the bound has been exceeded.

bool pos_sample_within_bound(vector<float> &weights, float currentWeight) {
  const bool debug = false;

  if (weights.size() == 1 && weights[0] > 0.5) {
    return false;
  }
  
  if (weights.size() <= 2) {
    // If there are not at least 3 values then don't bother trying to generate
    // a positive deltas window.
    return true;
  }

  vector<float> deltaWeights = float_diffs(weights);
  // Always ignore first element
  deltaWeights.erase(deltaWeights.begin());
  
  assert(deltaWeights.size() >= 2);
  
  int numNonNegDeltas = 0;
  
  vector<float> useDeltas;
  
  for (vector<float>::iterator it = deltaWeights.begin(); it != deltaWeights.end(); ++it) {
    float deltaWeight = *it;
    
    if (deltaWeight != 0.0f) {
      float absValue;
      if (deltaWeight > 0.0f) {
        absValue = deltaWeight;
        numNonNegDeltas += 1;
      } else {
        absValue = deltaWeight * -1;
      }
      useDeltas.push_back(absValue);
    }
  }
  
  if (debug) {
    cout << "abs deltas" << endl;
    for (vector<float>::iterator it = useDeltas.begin(); it != useDeltas.end(); ++it) {
      float delta = *it;
      cout << delta << endl;
    }
  }
  
  if (numNonNegDeltas >= 3) {
    if (debug) {
      cout << "will calculate pos delta window from only positive deltas" << endl;
    }
    
    useDeltas.erase(useDeltas.begin(), useDeltas.end());
    
    vector<float> increasingWeights;
    
    float prev = 0.0f; // will always be set in first iteration
    
    for (vector<float>::iterator it = weights.begin(); it != weights.end(); ++it) {
      if (it == weights.begin()) {
        prev = *it;
        continue;
      }
      
      float weight = *it;
      
      if (weight > prev) {
        // Only save weights that increase in value
        increasingWeights.push_back(weight);
        prev = weight;
      }
    }
    
    // FIXME: what happens if there is a big initial delte like 0.9 and then
    // the deltas after it are all smaller?
//
//    0.936088919
//    0.469772607
//    0.286601514
    
    assert(increasingWeights.size() > 0);
    
    if (debug) {
      cout << "increasingWeights" << endl;
      for (vector<float>::iterator it = increasingWeights.begin(); it != increasingWeights.end(); ++it) {
        float weight = *it;
        cout << weight << endl;
      }
    }
    
    vector<float> increasingDeltas = float_diffs(increasingWeights);
    // Always ignore first element since it is a delta from zero
    increasingDeltas.erase(increasingDeltas.begin());
    
    if (debug) {
      cout << "increasingDeltas" << endl;
      for (vector<float>::iterator it = increasingDeltas.begin(); it != increasingDeltas.end(); ++it) {
        float delta = *it;
        cout << delta << endl;
      }
    }
    
    // Save only positive weights and deltas
    
    weights = increasingWeights;
    useDeltas = increasingDeltas;
  } else {
    if (debug) {
      cout << "will calculate post delta window from abs deltas" << endl;
    }
  }
  
  float mean, stddev;
  
  sample_mean(useDeltas, &mean);
  sample_mean_delta_squared_div(useDeltas, mean, &stddev);
  
  float upperLimit = mean + (stddev * 2);
  
  float lastWeight = weights[weights.size()-1];
  
  float currentWeightDelta = currentWeight - lastWeight;

  if (debug) {
    cout << "mean " << mean << " stddev " << stddev << endl;
    
    cout << "1 stddev " << (mean + (stddev * 1)) << endl;
    cout << "2 stddev " << (mean + (stddev * 2)) << endl;
    cout << "3 stddev " << (mean + (stddev * 3)) << endl;
    
    cout << "last weight " << lastWeight << " currentWeight " << currentWeight << endl;
    cout << "currentWeightDelta " << currentWeightDelta << endl;
  }

  const float minStddev = 0.01f;

  if (stddev > minStddev && currentWeightDelta > 0.0f && currentWeightDelta > upperLimit) {
    
    if (debug) {
      cout << "stop expanding superpixel since currentWeightDelta > upperLimit : " << currentWeightDelta << " > " << upperLimit << endl;
    }
    
    return false;
  } else {
    
    if (debug) {
      if (stddev <= minStddev) {
        cout << "keep expanding superpixel since stddev <= minStddev : " << stddev << " <= " << minStddev << endl;
      } else {
        cout << "keep expanding superpixel since currentWeightDelta <= upperLimit : " << currentWeightDelta << " <= " << upperLimit << endl;
      }
    }
    
    return true;
  }
}

// Create a merge mask that shows the superpixel being considered and a graylevel of the neighbor
// superpixels to indicate the neighbor weigh. The first element of the merges vector indicates
// the superpixel ids considered in a merge step while the weights vector indicates the weights.

void writeSuperpixelMergeMask(SuperpixelImage &spImage, Mat &resultImg, vector<int32_t> merges, vector<float> weights, unordered_map<int32_t, bool> *lockedTablePtr)
{
  assert(merges.size() == weights.size());
  
  int wOffset = 0;
  
  // All locked superpixels as Red

  for (unordered_map<int32_t, bool>::iterator it = lockedTablePtr->begin(); it != lockedTablePtr->end(); ++it) {
    int32_t tag = it->first;
    
    Superpixel *spPtr = spImage.getSuperpixelPtr(tag);
    assert(spPtr);
    
    vector<pair<int32_t,int32_t> > &coords = spPtr->coords;
    
    for (vector<pair<int32_t,int32_t> >::iterator coordsIter = coords.begin(); coordsIter != coords.end(); ++coordsIter) {
      pair<int32_t,int32_t> coord = *coordsIter;
      int32_t X = coord.first;
      int32_t Y = coord.second;
      
      uint32_t pixel = 0xFFFF0000;
      
      Vec3b tagVec;
      tagVec[0] = pixel & 0xFF;
      tagVec[1] = (pixel >> 8) & 0xFF;
      tagVec[2] = (pixel >> 16) & 0xFF;
      resultImg.at<Vec3b>(Y, X) = tagVec;
    }
  }

  // Render weighted neighbors as grey values (inverted)
  
  for (vector<int32_t>::iterator it = merges.begin(); it != merges.end(); ++it) {
    int32_t tag = *it;
    
    bool isRootSperpixel = (it == merges.begin());
    float weight = weights[wOffset++];
    
    Superpixel *spPtr = spImage.getSuperpixelPtr(tag);
    assert(spPtr);
    
    vector<pair<int32_t,int32_t> > &coords = spPtr->coords;
    
    for (vector<pair<int32_t,int32_t> >::iterator coordsIter = coords.begin(); coordsIter != coords.end(); ++coordsIter) {
      pair<int32_t,int32_t> coord = *coordsIter;
      int32_t X = coord.first;
      int32_t Y = coord.second;
      
      uint32_t pixel;
      
      if (isRootSperpixel) {
        pixel = 0xFF00FF00;
      } else {
        uint32_t grey = int(round((1.0f - weight) * 255.0));
        pixel = (grey << 16) | (grey << 8) | grey;
      }
      
      Vec3b tagVec;
      tagVec[0] = pixel & 0xFF;
      tagVec[1] = (pixel >> 8) & 0xFF;
      tagVec[2] = (pixel >> 16) & 0xFF;
      resultImg.at<Vec3b>(Y, X) = tagVec;
    }
  }
}


// Gen a static colortable of a fixed size that contains enough colors to support
// the number of superpixels defined in the tags.

static vector<uint32_t> staticColortable;

// This table maps a superpixel tag to the offset in the staticColortable.

static unordered_map<int32_t,int32_t> staticTagToOffsetTable;

void generateStaticColortable(Mat &inputImg, SuperpixelImage &spImage)
{
  // Count number of superpixel to get max num colors
  
  int max = (int) spImage.superpixels.size();
  
  staticColortable.erase(staticColortable.begin(), staticColortable.end());
  
  for (int i = 0; i < max; i++) {
    uint32_t pixel = 0;
    pixel |= (rand() % 256);
    pixel |= ((rand() % 256) << 8);
    pixel |= ((rand() % 256) << 16);
    pixel |= (0xFF << 24);
    staticColortable.push_back(pixel);
  }
  
  // Lookup table from UID to offset into colortable
  
  staticTagToOffsetTable.erase(staticTagToOffsetTable.begin(), staticTagToOffsetTable.end());
  
  int32_t offset = 0;
  
  for (vector<int32_t>::iterator it = spImage.superpixels.begin(); it!=spImage.superpixels.end(); ++it) {
    int32_t tag = *it;
    staticTagToOffsetTable[tag] = offset;
    offset += 1;
  }
}

void writeTagsWithStaticColortable(SuperpixelImage &spImage, Mat &resultImg)
{
  for (vector<int32_t>::iterator it = spImage.superpixels.begin(); it != spImage.superpixels.end(); ++it) {
    int32_t tag = *it;
    
    Superpixel *spPtr = spImage.getSuperpixelPtr(tag);
    assert(spPtr);
    
    vector<pair<int32_t,int32_t> > &coords = spPtr->coords;
    
    for (vector<pair<int32_t,int32_t> >::iterator coordsIter = coords.begin(); coordsIter != coords.end(); ++coordsIter) {
      pair<int32_t,int32_t> coord = *coordsIter;
      int32_t X = coord.first;
      int32_t Y = coord.second;
      
      uint32_t offset = staticTagToOffsetTable[tag];
      uint32_t pixel = staticColortable[offset];
      
      Vec3b tagVec;
      tagVec[0] = pixel & 0xFF;
      tagVec[1] = (pixel >> 8) & 0xFF;
      tagVec[2] = (pixel >> 16) & 0xFF;
      resultImg.at<Vec3b>(Y, X) = tagVec;
    }
  }
}

// Assuming that there are N < 256 superpixels then the output can be writting as 8 bit grayscale.

void writeTagsWithGraytable(SuperpixelImage &spImage, Mat &origImg, Mat &resultImg)
{
  resultImg.create(origImg.rows, origImg.cols, CV_8UC(1));
  
  int gray = 0;
  
  // Sort superpixel by size
  
  vector<SuperpixelSortStruct> sortedSuperpixels;
  
  for (vector<int32_t>::iterator it = spImage.superpixels.begin(); it != spImage.superpixels.end(); ++it) {
    int32_t tag = *it;
    SuperpixelSortStruct ss;
    ss.spPtr = spImage.getSuperpixelPtr(tag);
    sortedSuperpixels.push_back(ss);
  }
  
  sort(sortedSuperpixels.begin(), sortedSuperpixels.end(), CompareSuperpixelSizeDecreasingFunc);
  
  for (vector<SuperpixelSortStruct>::iterator it = sortedSuperpixels.begin(); it != sortedSuperpixels.end(); ++it) {
    SuperpixelSortStruct ss = *it;
    Superpixel *spPtr = ss.spPtr;
    assert(spPtr);
    
    //cout << "N = " << (int)spPtr->coords.size() << endl;
    
    vector<pair<int32_t,int32_t> > &coords = spPtr->coords;
    
    for (vector<pair<int32_t,int32_t> >::iterator coordsIter = coords.begin(); coordsIter != coords.end(); ++coordsIter) {
      pair<int32_t,int32_t> coord = *coordsIter;
      int32_t X = coord.first;
      int32_t Y = coord.second;
      
      resultImg.at<uint8_t>(Y, X) = gray;
    }
    
    gray++;
  }
}

// Generate gray table and the write pixels as int BGR.

void writeTagsWithMinColortable(SuperpixelImage &spImage, Mat &origImg, Mat &resultImg)
{
  resultImg.create(origImg.rows, origImg.cols, CV_8UC(3));
  
  int gray = 0;
  
  // Sort superpixel by size
  
  vector<SuperpixelSortStruct> sortedSuperpixels;
  
  for (vector<int32_t>::iterator it = spImage.superpixels.begin(); it != spImage.superpixels.end(); ++it) {
    int32_t tag = *it;
    SuperpixelSortStruct ss;
    ss.spPtr = spImage.getSuperpixelPtr(tag);
    sortedSuperpixels.push_back(ss);
  }
  
  sort(sortedSuperpixels.begin(), sortedSuperpixels.end(), CompareSuperpixelSizeDecreasingFunc);
  
  for (vector<SuperpixelSortStruct>::iterator it = sortedSuperpixels.begin(); it != sortedSuperpixels.end(); ++it) {
    SuperpixelSortStruct ss = *it;
    Superpixel *spPtr = ss.spPtr;
    assert(spPtr);
    
    //cout << "N[" << gray << "] = " << (int)spPtr->coords.size() << endl;
    
    vector<pair<int32_t,int32_t> > &coords = spPtr->coords;
    
    for (vector<pair<int32_t,int32_t> >::iterator coordsIter = coords.begin(); coordsIter != coords.end(); ++coordsIter) {
      pair<int32_t,int32_t> coord = *coordsIter;
      int32_t X = coord.first;
      int32_t Y = coord.second;
      
      uint8_t B = gray & 0xFF;
      uint8_t G = (gray >> 8) & 0xFF;
      uint8_t R = (gray >> 16) & 0xFF;
      
      Vec3b pixelVec(B,G,R);
      
      resultImg.at<Vec3b>(Y, X) = pixelVec;
    }
    
    gray++;
  }
}




