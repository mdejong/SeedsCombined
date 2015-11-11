// A superpixel image is a matrix that contains N superpixels and N superpixel edges between superpixels.
// A superpixel image is typically parsed from a source of tags, modified, and then written as a new tags
// image.

#ifndef SUPERPIXEL_IMAGE_H
#define	SUPERPIXEL_IMAGE_H

#include <opencv2/opencv.hpp>

#include <unordered_map>

using namespace std;
using namespace cv;

class Superpixel;
class SuperpixelEdge;

#include "SuperpixelEdgeTable.h"

typedef unordered_map<int32_t, Superpixel*> TagToSuperpixelMap;

typedef tuple<double, int32_t, int32_t> CompareNeighborTuple;

typedef enum {
  BACKPROJECT_HIGH_FIVE, // top 95% with gray = 200
  BACKPROJECT_HIGH_FIVE8, // top 95% with gray = 200 (8 bins per channel)
  BACKPROJECT_HIGH_TEN,  // top 90% with gray 200
  BACKPROJECT_HIGH_15,  // top 85% with gray 200
  BACKPROJECT_HIGH_20,  // top 80% with gray 200
  BACKPROJECT_HIGH_50,  // top 80% with gray 200
} BackprojectRange;


class SuperpixelImage {
  
  public:
  
  // This map contains the actual pointers to Superpixel objects.
  
  TagToSuperpixelMap tagToSuperpixelMap;
  
  // The superpixels list contains the UIDs for superpixels
  // in UID sorted order.
  
  vector<int32_t> superpixels;

  // The edge table contains a representation of "edges" in terms
  // of adjacent nodes lists.
  
  SuperpixelEdgeTable edgeTable;
  
  // This superpixel edge merge order list is only active in DEBUG.

#if defined(DEBUG)
  vector<SuperpixelEdge> mergeOrder;
#endif
  
  // Lookup Superpixel* given a UID

  Superpixel* getSuperpixelPtr(int32_t uid);
  
  // Parse tags image and construct superpixels. Note that this method will modify the
  // original tag values by adding 1 to each original tag value.
  
  static
  bool parse(Mat &tags, SuperpixelImage &spImage);

  static
  bool parseSuperpixelEdges(Mat &tags, SuperpixelImage &spImage);
  
  // Return vector of all edges
  vector<SuperpixelEdge> getEdges();
  
  // Merge superpixels defined by edge in this image container
  
  void mergeEdge(SuperpixelEdge &edge);
  
  // Merge superpixels where all pixels are the same pixel.
  
  void mergeIdenticalSuperpixels(Mat &inputImg);

  int mergeBackprojectSuperpixels(Mat &inputImg, int colorspace, int startStep, BackprojectRange range);

  // Merge small superpixels away from the largest neighbor.
  
  int mergeSmallSuperpixels(Mat &inputImg, int colorspace, int startStep);
  
  // Merge superpixels detected as "edges" away from the largest neighbor.

  int mergeEdgySuperpixels(Mat &inputImg, int colorspace, int startStep, vector<int32_t> *largeSuperpixelsPtr);
  
  void scanLargestSuperpixels(vector<int32_t> &results);
  
  void rescanLargestSuperpixels(Mat &inputImg, Mat &outputImg, vector<int32_t> *largeSuperpixelsPtr);
  
  void fillMatrixFromCoords(Mat &input, int32_t tag, Mat &output);
  
  // This method is the inverse of fillMatrixFromCoords(), it reads pixel values from a matrix
  // and writes them back to X,Y values that correspond to the original image. This method is
  // very useful when running an image operation on all the pixels in a superpixel but without
  // having to process all the pixels in a bbox area. The dimensions of the input must be
  // NUM_COORDS x 1.
  
  void reverseFillMatrixFromCoords(Mat &input, bool isGray, int32_t tag, Mat &output);
  
  // true when all pixels in a superpixel are exactly identical
  
  bool isAllSamePixels(Mat &input, int32_t tag);
  
  // When a superpixel is known to have all identical pixel values then only the first
  // pixel in that superpixel needs to be compared to all the other pixels in a second
  // superpixel.
  
  bool isAllSamePixels(Mat &input, Superpixel *spPtr, int32_t otherTag);
  
  bool isAllSamePixels(Mat &input, uint32_t knownFirstPixel, vector<pair<int32_t,int32_t> > &coords);
  
  // Compare function that does histogram compare for each neighbor of superpixel tag
  
  void compareNeighborSuperpixels(Mat &inputImg,
                                  int32_t tag,
                                  vector<CompareNeighborTuple> &results,
                                  unordered_map<int32_t, bool> *lockedTablePtr,
                                  int32_t step);
  
  // Compare function that examines neighbor edges
  
  void compareNeighborEdges(Mat &inputImg,
                            int32_t tag,
                            vector<CompareNeighborTuple> &results,
                            unordered_map<int32_t, bool> *lockedTablePtr,
                            int32_t step,
                            bool normalize);
  
  // Evaluate backprojection of superpixel to the connected neighbors

  void backprojectNeighborSuperpixels(Mat &inputImg,
                                      int32_t tag,
                                      vector<CompareNeighborTuple> &results,
                                      unordered_map<int32_t, bool> *lockedTablePtr,
                                      int32_t step,
                                      int conversion,
                                      int numPercentRanges,
                                      int numTopPercent,
                                      bool roundPercent,
                                      int minGraylevel,
                                      int numBins);
  
  void backprojectDepthFirstRecurseIntoNeighbors(Mat &inputImg,
                                                 int32_t tag,
                                                 vector<int32_t> &results,
                                                 unordered_map<int32_t, bool> *lockedTablePtr,
                                                 int32_t step,
                                                 int conversion,
                                                 int numPercentRanges,
                                                 int numTopPercent,
                                                 int minGraylevel,
                                                 int numBins);
  
  // Recursive bredth first search to fully expand the largest superpixel in a BFS order
  // and then lock the superpixel before expanding in terms of smaller superpixels. This
  // logic looks for possible expansion using back projection but it keeps track of
  // edge weights so that an edge will not be collapsed when it has a very high weight
  // as compared to the other edge weights for this specific superpixel.
  
  int mergeBredthFirst(Mat &inputImg, int colorspace, int startStep, vector<int32_t> *largeSuperpixelsPtr, int numBins);
  
  void applyBilinearFiltering(Mat &inputImg, Mat &outputImg, int kernel, int cradius);
  
  void filterOutVeryLargeNeighbors(int32_t tag, vector<int32_t> &neighbors);
  
  bool shouldMergeEdge(int32_t tag, float edgeWeight);
  
  void addUnmergedEdgeWeights(int32_t tag, vector<float> &edgeWeights);
  
  void addMergedEdgeWeight(int32_t tag, float edgeWeight);
  
  void checkNeighborEdgeWeights(Mat &inputImg,
                                int32_t tag,
                                vector<int32_t> *neighborsPtr,
                                unordered_map<SuperpixelEdge, float> &edgeStrengthMap,
                                int step);
  
  void sortSuperpixelsBySize();
};

#endif // SUPERPIXEL_IMAGE_H
