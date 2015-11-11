/**
 * Command line tool for using SEEDS Revised, an implementation of the superpixel
 * algorithm proposed in [1] and evaluated in [2].
 * 
 *  [1] M. van den Bergh, X. Boix, G. Roig, B. de Capitani, L. van Gool.
 *      SEEDS: Superpixels extracted via energy-driven sampling.
 *      Proceedings of the European Conference on Computer Vision, pages 13–26, 2012.
 *  [2] D. Stutz, A. Hermans, B. Leibe.
 *      Superpixel Segmentation using Depth Information.
 *      Bachelor thesis, RWTH Aachen University, Aachen, Germany, 2014.
 * 
 * [2] is available online at 
 * 
 *      http://davidstutz.de/bachelor-thesis-superpixel-segmentation-using-depth-information/
 * 
 * **How to use the command line tool?**
 * 
 * Compile both the library in `/lib` as well as `cli/main.cpp` using CMake (see
 * `README.md`). The provided options can be viewed using 
 * 
 *  $ ./bin/cli --help
 *  Allowed options:
 *   --help                          produce help message
 *   --input arg                     the folder to process, may contain several 
 *                                   images
 *   --bins arg (=5)                 number of bins used for color histograms
 *   --neighborhood arg (=1)         neighborhood size used for smoothing prior
 *   --confidence arg (=0.100000001) minimum confidence used for block update
 *   --iterations arg (=2)           iterations at each level
 *   --spatial-weight arg (=0.25)    spatial weight
 *   --superpixels arg (=400)        desired number of supüerpixels
 *   --process                       show additional information while processing
 *   --csv                           save segmentation as CSV file
 *   --contour                       save contour image of segmentation
 *   --mean                          save mean colored image of segmentation
 *   --output arg (=output)          specify the output directory (default is 
 *                                   ./output)
 * 
 * The code is published under the BSD 3-Clause:
 * 
 * Copyright (c) 2014, David Stutz
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "SeedsRevised.h"
#include "Tools.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, const char** argv) {
  if (argc != 2 && argc != 3) {
    cerr << "usage : " << argv[0] << " INPUT_PNG ?NUM_SPIXELS?" << endl;
    exit(1);
  }
  char *inPNG;
  int numSuperpixels = -1;
  
  if (argc == 2 || argc == 3) {
    inPNG = (char*) argv[1];
  }
  if (argc == 3) {
    numSuperpixels = atoi((char*) argv[2]);
  }
  
    cout << "reading image " << inPNG << endl;
    
    int iterations = 2; // iterations at each level
    int numberOfBins = 5; // number of bins used for color histograms
    int neighborhoodSize = 1; // neighborhood size used for smoothing prior
    float minimumConfidence = 0.1; // minimum confidence used for block update;
    float spatialWeight = 0.25;
    
    Mat image = imread(inPNG, CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    cerr << "image could not be read " << inPNG << endl;
    exit(1);
  }
  
  if (numSuperpixels == -1) {
    // Calculate the number of superpixels to use as an upper limit of the rational size
    // that an image could be divided into with very small 10x10 or 5x5 superpixels.
    // It is okay to have this N be too large since the limit of the oversegmentation
    // process will only reduce pixels down so far. While faster segmentation is possible
    // the goal of having a small superpixel is more important for good segmentation with
    // this N value.
    
    int width = image.rows;
    int height = image.cols;
    
    int dims = width * height;
    
    if (dims < 50*50) {
      numSuperpixels = 500;
    } else if (dims < 200*200) {
      numSuperpixels = 1000;
    } else if (dims < 1000*1000) {
      numSuperpixels = 5000;
    } else {
      numSuperpixels = 8000;
    }
    
    cout << "auto N superpixels = " << numSuperpixels << endl;
  }
  
    int superpixels = numSuperpixels; // num superpixels
  
    SEEDSRevisedMeanPixels seeds(image, superpixels, numberOfBins, neighborhoodSize, minimumConfidence, spatialWeight);

    seeds.initialize();
    seeds.iterate(iterations);

        if (1) {
            string store = "contours.png";

            int bgr[] = {0, 0, 204};
            Mat contourImage = Draw::contourImage(seeds.getLabels(), image, bgr);
            imwrite(store, contourImage);

            cout << "Image with contours saved to " << store << " ..." << endl;
        }

  if (1) {
    string store = "edges.png";
    
    int bgr[] = {255, 255, 255};
    Mat edgesGray = image.clone();
    
    edgesGray.setTo(Scalar(0,0,0));
    
    Mat edgesGray2 = Draw::contourImage(seeds.getLabels(), edgesGray, bgr);
    
    Mat edgesGray3;
    cvtColor(edgesGray2, edgesGray3, CV_RGB2GRAY);
    
    imwrite(store, edgesGray3);
    
    cout << "Image with only edges saved to " << store << " ..." << endl;
  }
  
        if (1) {
            string store = "mean.png";

            Mat meanImage = Draw::meanImage(seeds.getLabels(), image);
            imwrite(store, meanImage);

            cout << "Image with mean colors saved to " << store << " ..." << endl;
        }

  if (1) {
    string store = "clusters.png";
    
    Integrity::relabel(seeds.getLabels(), image.rows, image.cols);
    
    Mat clustersRGB = Draw::labelImage(seeds.getLabels(), image);
    
    imwrite(store, clustersRGB);
    
    cout << "Image with cluser ids saved to " << store << " ..." << endl;
  }
  
    return 0;
}
