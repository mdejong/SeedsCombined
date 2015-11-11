// This file contains utility functions for general purpose use and interaction with OpenCV

#include "OpenCVUtil.h"

// Print SSIM for two images to cout

int printSSIM(Mat inImage1, Mat inImage2)
{
  // default settings
  double C1 = 6.5025, C2 = 58.5225;
  
  IplImage
		*img1=NULL, *img2=NULL, *img1_img2=NULL,
		*img1_temp=NULL, *img2_temp=NULL,
		*img1_sq=NULL, *img2_sq=NULL,
		*mu1=NULL, *mu2=NULL,
		*mu1_sq=NULL, *mu2_sq=NULL, *mu1_mu2=NULL,
		*sigma1_sq=NULL, *sigma2_sq=NULL, *sigma12=NULL,
		*ssim_map=NULL, *temp1=NULL, *temp2=NULL, *temp3=NULL;

  IplImage tmpCopy1 = inImage1;
  IplImage tmpCopy2 = inImage2;
  
  img1_temp = &tmpCopy1;
  img2_temp = &tmpCopy2;
  
  int x=img1_temp->width, y=img1_temp->height;
  int nChan=img1_temp->nChannels, d=IPL_DEPTH_32F;
  CvSize size = cvSize(x, y);
  
  img1 = cvCreateImage(size, d, nChan);
  img2 = cvCreateImage(size, d, nChan);
  
  cvConvert(img1_temp, img1);
  cvConvert(img2_temp, img2);
  cvReleaseImage(&img1_temp);
  img1_temp = NULL;
  cvReleaseImage(&img2_temp);
  img2_temp = NULL;
  
  img1_sq = cvCreateImage( size, d, nChan);
  img2_sq = cvCreateImage( size, d, nChan);
  img1_img2 = cvCreateImage( size, d, nChan);
  
  cvPow( img1, img1_sq, 2 );
  cvPow( img2, img2_sq, 2 );
  cvMul( img1, img2, img1_img2, 1 );
  
  mu1 = cvCreateImage( size, d, nChan);
  mu2 = cvCreateImage( size, d, nChan);
  
  mu1_sq = cvCreateImage( size, d, nChan);
  mu2_sq = cvCreateImage( size, d, nChan);
  mu1_mu2 = cvCreateImage( size, d, nChan);
  
  
  sigma1_sq = cvCreateImage( size, d, nChan);
  sigma2_sq = cvCreateImage( size, d, nChan);
  sigma12 = cvCreateImage( size, d, nChan);
  
  temp1 = cvCreateImage( size, d, nChan);
  temp2 = cvCreateImage( size, d, nChan);
  temp3 = cvCreateImage( size, d, nChan);
  
  ssim_map = cvCreateImage( size, d, nChan);
  
  // ************************** END INITS **********************************
  
  //////////////////////////////////////////////////////////////////////////
  // PRELIMINARY COMPUTING
  cvSmooth( img1, mu1, CV_GAUSSIAN, 11, 11, 1.5 );
  cvSmooth( img2, mu2, CV_GAUSSIAN, 11, 11, 1.5 );
  
  cvPow( mu1, mu1_sq, 2 );
  cvPow( mu2, mu2_sq, 2 );
  cvMul( mu1, mu2, mu1_mu2, 1 );
  
  
  cvSmooth( img1_sq, sigma1_sq, CV_GAUSSIAN, 11, 11, 1.5 );
  cvAddWeighted( sigma1_sq, 1, mu1_sq, -1, 0, sigma1_sq );
  
  cvSmooth( img2_sq, sigma2_sq, CV_GAUSSIAN, 11, 11, 1.5 );
  cvAddWeighted( sigma2_sq, 1, mu2_sq, -1, 0, sigma2_sq );
  
  cvSmooth( img1_img2, sigma12, CV_GAUSSIAN, 11, 11, 1.5 );
  cvAddWeighted( sigma12, 1, mu1_mu2, -1, 0, sigma12 );
  
  
  //////////////////////////////////////////////////////////////////////////
  // FORMULA
  
  // (2*mu1_mu2 + C1)
  cvScale( mu1_mu2, temp1, 2 );
  cvAddS( temp1, cvScalarAll(C1), temp1 );
  
  // (2*sigma12 + C2)
  cvScale( sigma12, temp2, 2 );
  cvAddS( temp2, cvScalarAll(C2), temp2 );
  
  // ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
  cvMul( temp1, temp2, temp3, 1 );
  
  // (mu1_sq + mu2_sq + C1)
  cvAdd( mu1_sq, mu2_sq, temp1 );
  cvAddS( temp1, cvScalarAll(C1), temp1 );
  
  // (sigma1_sq + sigma2_sq + C2)
  cvAdd( sigma1_sq, sigma2_sq, temp2 );
  cvAddS( temp2, cvScalarAll(C2), temp2 );
  
  // ((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
  cvMul( temp1, temp2, temp1, 1 );
  
  // ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
  cvDiv( temp3, temp1, ssim_map, 1 );
  
  CvScalar index_scalar = cvAvg( ssim_map );
  
  // through observation, there is approximately
  // 1% error max with the original matlab program
  
  cout << "(R, G & B SSIM index)" << endl ;
  cout << index_scalar.val[2] * 100 << "%" << endl ;
  cout << index_scalar.val[1] * 100 << "%" << endl ;
  cout << index_scalar.val[0] * 100 << "%" << endl ;
  
  // Release the images
  
  if (img1) {
    cvReleaseImage(&img1);
  }
  if (img2) {
    cvReleaseImage(&img2);
  }
  if (img1_img2) {
    cvReleaseImage(&img1_img2);
  }
  if (img1_temp) {
    cvReleaseImage(&img1_temp);
  }
  if (img2_temp) {
    cvReleaseImage(&img2_temp);
  }
  if (img1_sq) {
    cvReleaseImage(&img1_sq);
  }
  if (img2_sq) {
    cvReleaseImage(&img2_sq);
  }
  if (mu1) {
    cvReleaseImage(&mu1);
  }
  if (mu2) {
    cvReleaseImage(&mu2);
  }
  if (mu1_sq) {
    cvReleaseImage(&mu1_sq);
  }
  if (mu2_sq) {
    cvReleaseImage(&mu2_sq);
  }
  if (mu1_mu2) {
    cvReleaseImage(&mu1_mu2);
  }
  if (sigma1_sq) {
    cvReleaseImage(&sigma1_sq);
  }
  if (sigma2_sq) {
    cvReleaseImage(&sigma2_sq);
  }
  if (sigma12) {
    cvReleaseImage(&sigma12);
  }
  if (ssim_map) {
    cvReleaseImage(&ssim_map);
  }
  if (temp1) {
    cvReleaseImage(&temp1);
  }
  if (temp2) {
    cvReleaseImage(&temp2);
  }
  if (temp3) {
    cvReleaseImage(&temp3);
  }
  
  return 0;
}
