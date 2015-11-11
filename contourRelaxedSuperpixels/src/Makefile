OPENCV_INSTALL=/Users/modejong/Development/OpenCV/install
INCLUDES=-I$(OPENCV_INSTALL)/include -I$(OPENCV_INSTALL)/include/opencv

LIBS2=-L$(OPENCV_INSTALL)/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -lopencv_photo -lobjc -L$(OPENCV_INSTALL)/share/OpenCV/3rdparty/lib -l libpng -l libjpeg -l libjasper -l zlib -framework AppKit -framework QuartzCore -framework QTKit -framework Cocoa -lstdc++ -framework OpenCL -lIlmImf -llibtiff -lopencv_legacy

LIBS3=-L$(OPENCV_INSTALL)/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_objdetect -lopencv_photo -lobjc -L$(OPENCV_INSTALL)/share/OpenCV/3rdparty/lib -l libpng -l libjpeg -l libjasper -l zlib -framework AppKit -framework QuartzCore -framework QTKit -framework Cocoa -lstdc++ -framework OpenCL -lIlmImf -llibwebp -llibtiff

LIBS=$(LIBS2)

all: contourRelaxedSuperpixels

contourRelaxedSuperpixels: contourRelaxedSuperpixels.cpp
	clang -o contourRelaxedSuperpixels -g contourRelaxedSuperpixels.cpp $(INCLUDES) $(LIBS)

