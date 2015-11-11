// seedscombine IMAGE TAGS_IMAGE OUT_TAGS
//
// Given the original input image and an oversegmented image merge superpixels in terms of alike colors
// and generate an output tags image that significantly simplifies the superpixel segmentation.
// This logic makes use of histogram backprojection to scan pixels quickly for alikeness.

#include <opencv2/opencv.hpp>

#include "Superpixel.h"
#include "SuperpixelEdge.h"
#include "SuperpixelImage.h"

using namespace cv;
using namespace std;

bool seedsCombine(Mat &inputImg, Mat &tagsImg, Mat &resultImg);

void generateStaticColortable(Mat &inputImg, SuperpixelImage &spImage);

void writeTagsWithStaticColortable(SuperpixelImage &spImage, Mat &resultImg);

void writeTagsWithGraytable(SuperpixelImage &spImage, Mat &origImg, Mat &resultImg);

void writeTagsWithMinColortable(SuperpixelImage &spImage, Mat &origImg, Mat &resultImg);

int main(int argc, const char** argv) {
  if (argc != 4) {
    cerr << "usage : " << argv[0] << " IMAGE TAGS_IMAGE OUT_TAGS" << endl;
    exit(1);
  }
  
  const char *inputImgFilename = argv[1];
  const char *tagsImgFilename = argv[2];
  const char *outputTagsImgFilename = argv[3];

  Mat inputImg = imread(inputImgFilename, CV_LOAD_IMAGE_COLOR);
  if( inputImg.empty() ) {
    cerr << "could not read \"" << inputImgFilename << "\" as image data" << endl;
    exit(1);
  }
  
  Mat tagsImg = imread(tagsImgFilename, CV_LOAD_IMAGE_COLOR);
  if( tagsImg.empty() ) {
    cerr << "could not read \"" << tagsImgFilename << "\" as image data" << endl;
    exit(1);
  }
  
  // The width x height of image and tags must be identical
  
  if ((inputImg.rows != tagsImg.rows) || (inputImg.cols != tagsImg.cols)) {
    cerr << "input image and tags dimensions must match "
      << inputImg.cols << " x " << inputImg.rows
      << " != "
      << tagsImg.cols << " x " << tagsImg.rows << endl;
    exit(1);
  }
  
  Mat resultImg;
  
  bool worked = seedsCombine(inputImg, tagsImg, resultImg);
  if (!worked) {
    cerr << "seeds combine failed " << endl;
    exit(1);
  }
  
  imwrite(outputTagsImgFilename, resultImg);
  
  cout << "wrote " << outputTagsImgFilename << endl;
  
  exit(0);
}

bool seedsCombine(Mat &inputImg, Mat &tagsImg, Mat &resultImg)
{
  const bool debugWriteIntermediateFiles = true;
  
  SuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(tagsImg, spImage);
  
  if (!worked) {
    return false;
  }
  
  // Dump image that shows the input superpixels written with a colortable
  
  resultImg = inputImg.clone();
  resultImg = (Scalar) 0;

  sranddev();
  
  if (debugWriteIntermediateFiles) {
    generateStaticColortable(inputImg, spImage);
  }

  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_input.png", resultImg);
  }
  
  cout << "started with " << spImage.superpixels.size() << " superpixels" << endl;
  
  // Identical
  
  spImage.mergeIdenticalSuperpixels(inputImg);
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_after_identical_merge.png", resultImg);
  }
  
  int mergeStep = 0;
  
  // RGB
  
  mergeStep = spImage.mergeBackprojectSuperpixels(inputImg, 0, mergeStep, BACKPROJECT_HIGH_FIVE);
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_after_backproject_merge_RGB.png", resultImg);
  }
  
  cout << "RGB merge edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
  
  // LAB
  
  mergeStep = spImage.mergeBackprojectSuperpixels(inputImg, CV_BGR2Lab, mergeStep, BACKPROJECT_HIGH_FIVE);
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_after_backproject_merge_LAB.png", resultImg);
  }
  
  cout << "LAB merge edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
  
  // After the initial processing above, identical and very alike regions have been joined as blobs.
  // The next stage does back projection again but in away that compares edge weights between the
  // superpixels to detect when to stop searching for histogram alikeness.
  
  mergeStep = spImage.mergeBredthFirst(inputImg, 0, mergeStep, NULL, 16);
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    
    imwrite("tags_after_backproject_fill_merge_RGB.png", resultImg);
  }
  
  cout << "RGB fill backproject merge edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;

  // Select only small superpixels and then merge away from largest neighbors
  
  mergeStep = spImage.mergeSmallSuperpixels(inputImg, 0, mergeStep);
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_after_small_merge_RGB.png", resultImg);
  }
  
  cout << "RGB merge small count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;

  // Get large superpixels list at this point. It is generally cleaned up and closer to well
  // defined hard edges.
  
  vector<int32_t> veryLargeSuperpixels;
  spImage.scanLargestSuperpixels(veryLargeSuperpixels);

  // Merge edgy superpixel into edgy neighbor(s)

  mergeStep = spImage.mergeEdgySuperpixels(inputImg, 0, mergeStep, &veryLargeSuperpixels);
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_after_edgy_RGB.png", resultImg);
  }
  
  cout << "RGB merge edgy count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
  
  // LAB
  
  mergeStep = spImage.mergeBackprojectSuperpixels(inputImg, CV_BGR2Lab, mergeStep, BACKPROJECT_HIGH_FIVE);
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_after_cleanup_backproject_merge_LAB.png", resultImg);
  }
  
  cout << "LAB merge count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
  
  if (spImage.superpixels.size() < 256) {
    Mat grayImg;
    writeTagsWithGraytable(spImage, inputImg, grayImg);
    imwrite("tags_grayscale.png", grayImg);
    
    cout << "wrote " << "tags_grayscale.png" << endl;
  }
  
  Mat minImg;

  writeTagsWithMinColortable(spImage, inputImg, minImg);
  imwrite("tags_min_color.png", minImg);
  cout << "wrote " << "tags_min_color.png" << endl;
  
  // Done
  
  cout << "ended with " << spImage.superpixels.size() << " superpixels" << endl;
  
  return true;
}

