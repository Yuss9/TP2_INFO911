#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;


/* 
###################################################################
*/

Mat filtreM(Mat input, Mat M) {
    Mat result = Mat::zeros(input.size(), input.type());

    int mRows = M.rows;
    int mCols = M.cols;
    int mCenterX = mCols / 2;
    int mCenterY = mRows / 2;

    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            double sum = 0.0;

            for (int m = 0; m < mRows; ++m) {
                int mm = mRows - 1 - m;

                for (int n = 0; n < mCols; ++n) {
                    int nn = mCols - 1 - n;

                    int ii = i + m - mCenterY;
                    int jj = j + n - mCenterX;

                    if (ii >= 0 && ii < input.rows && jj >= 0 && jj < input.cols) {
                        sum += input.at<uchar>(ii, jj) * M.at<float>(mm, nn);
                    }
                }
            }

            result.at<uchar>(i, j) = saturate_cast<uchar>(sum);
        }
    }

    return result;
}

void test_filtre_moyenneur(String filename){
    namedWindow("Filtre moyenneur"); 
    Mat input = imread(filename); 
    Mat M = (Mat_<float>(3, 3) << 1.0 / 16, 2.0 / 16, 1.0 / 16,
                                        2.0 / 16, 4.0 / 16, 2.0 / 16,
                                        1.0 / 16, 2.0 / 16, 1.0 / 16);

    if (input.channels() == 3)
        cvtColor(input, input, COLOR_BGR2GRAY);

    while (true) {
        int keycode = waitKey(50);
        int asciicode = keycode & 0xff;

        if (asciicode == 'q')
            break;
        else if (asciicode == 'a') {
            for (int i = 0; i < 100; i++)
            {
                input = filtreM(input, M);
            }
            imshow("Filtre moyenneur", input);
        } else {
            imshow("Filtre moyenneur", input);
        }
    }

    imwrite("result.png", input);
}

/* 
###################################################################
*/


Mat filtreMedian(Mat input) {
    Mat result = input.clone();

    int mRows = 3;
    int mCols = 3;
    int mCenterX = mCols / 2;
    int mCenterY = mRows / 2;

    for (int i = mCenterY; i < input.rows - mCenterY; ++i) {
        for (int j = mCenterX; j < input.cols - mCenterX; ++j) {
            vector<uchar> values;

            for (int m = 0; m < mRows; ++m) {
                for (int n = 0; n < mCols; ++n) {
                    int ii = i + m - mCenterY;
                    int jj = j + n - mCenterX;
                    values.push_back(input.at<uchar>(ii, jj));
                }
            }

            sort(values.begin(), values.end());
            result.at<uchar>(i, j) = values[values.size() / 2];
        }
    }

    return result;
}

// avec la fonction d'opencv pour comparer
Mat filtreMedianBlur(Mat input) {
    Mat result;
    medianBlur(input, result, 3); // Taille du noyau : 3x3
    return result;
}

void test_filtre_median(String filename){
    Mat input = imread(filename);
    Mat input_opencv = imread(filename);

    if (input.channels() == 3)
        cvtColor(input, input, COLOR_BGR2GRAY);
        cvtColor(input_opencv, input_opencv, COLOR_BGR2GRAY);

    while (true) {
        int keycode = waitKey(50);
        int asciicode = keycode & 0xff;

        if (asciicode == 'q')
            break;
        else if (asciicode == 'm') {

            for (int i = 0; i < 16; i++)
            {
                input = filtreMedianBlur(input);
                input_opencv = filtreMedianBlur(input_opencv);
            }
            
            imshow("Image self made", input);
            imshow("Image opencv function", input_opencv);
        } else {
            imshow("Image self made", input);
            imshow("Image opencv function", input_opencv);
        }
    }

    imwrite("result.png", input);
    imwrite("result_opencv.png", input_opencv);

}

/* 
###################################################################
*/


int alpha = 0.6; 

void onTrackbarChange(int, void*) {
    // Cette fonction sera appelée à chaque changement de la position du slider
    // Elle met à jour la variable alpha
    alpha = getTrackbarPos("alpha (en %)", "Image");
}

Mat rehaussementContraste(Mat input, int alpha_i) {
    float alpha = alpha_i / 100.0;
    cv::Mat laplacien = (Mat_<float>(3,3) << 0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0);
    cv:Mat dirac = (Mat_<float>(3,3) << 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);

    Mat R = dirac - alpha * laplacien;

    return filtreM(input, R);
}



void test_rehausse_contraste(String filename){
    namedWindow( "Filter");              
    int alpha = 20;
    createTrackbar( "alpha (en %)", "Filter", &alpha, 100,  NULL);
    setTrackbarPos( "alpha (en %)", "Filter", alpha ); // init à 20

    Mat input = imread( filename );     // lit l'image donnée en paramètre
    if ( input.channels() == 3 )
        cv::cvtColor( input, input, COLOR_BGR2GRAY );
    
    while ( true ) {

        cv::Mat moyenneur = (Mat_<float>(3,3) << 1.0/16.0f, 2.0/16.0f, 1.0/16.0f, 2.0/16.0f, 4.0/16.0f, 2.0/16.0f, 1.0/16.0f, 2.0/16.0f, 1.0/16.0f);
        

        int keycode = waitKey( 50 );
        int asciicode = keycode & 0xff;
        if ( asciicode == 'q' ) break;
        if ( asciicode == 'M') input = filtreM(input, moyenneur);
        if ( asciicode == 'm' ) for(int i = 0; i < 100; i++) cv::medianBlur(input, input, 3);
        if ( asciicode == 's' ) input = rehaussementContraste(input, getTrackbarPos( "alpha (en %)", "Filter")); 
        imshow( "Filter", input );           
    }

    imwrite( "result.png", input );          
}

/* 
###################################################################
*/

cv::Mat sobelFilter(const cv::Mat& input, bool isVertical) {
    cv::Mat result(input.size(), CV_32F, Scalar(0));

    int rows = input.rows;
    int cols = input.cols;

    int kernelSize = 3;
    int sobelKernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

    if (!isVertical) {
        kernelSize = 3;
        int tempKernel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        memcpy(sobelKernel, tempKernel, sizeof(int) * 3 * 3);
    }

    int kernelCenter = kernelSize / 2;

    for (int i = kernelCenter; i < rows - kernelCenter; ++i) {
        for (int j = kernelCenter; j < cols - kernelCenter; ++j) {
            float sum = 0.0;
            for (int u = 0; u < kernelSize; ++u) {
                for (int v = 0; v < kernelSize; ++v) {
                    sum += input.at<uchar>(i + u - kernelCenter, j + v - kernelCenter) * sobelKernel[u][v];
                }
            }
            result.at<float>(i, j) = sum;
        }
    }

    return result;
}

void test_sobel(String filename){
    Mat input = imread(filename, IMREAD_GRAYSCALE);

    if (input.empty()) {
        cerr << "Error: Couldn't load the image." << endl;
        return;
    }

    namedWindow("Original Image", WINDOW_NORMAL);
    imshow("Original Image", input);

    Mat sobelVertical = sobelFilter(input, true);
    Mat sobelHorizontal = sobelFilter(input, false);

    // Shift the result to have zero at 128
    sobelVertical += 128;
    sobelHorizontal += 128;

    sobelVertical.convertTo(sobelVertical, CV_8U);
    sobelHorizontal.convertTo(sobelHorizontal, CV_8U);

    namedWindow("Sobel Vertical", WINDOW_NORMAL);
    imshow("Sobel Vertical", sobelVertical);

    namedWindow("Sobel Horizontal", WINDOW_NORMAL);
    imshow("Sobel Horizontal", sobelHorizontal);

    waitKey(0);
}


/* 
###################################################################
*/

cv::Mat calculateGradient(const cv::Mat& Ix, const cv::Mat& Iy) {
    cv::Mat gradientMagnitude(Ix.size(), CV_32F);
    for (int i = 0; i < Ix.rows; ++i) {
        for (int j = 0; j < Ix.cols; ++j) {
            float ix = Ix.at<float>(i, j);
            float iy = Iy.at<float>(i, j);

            float magnitude = std::sqrt(pow(ix,2) + pow(iy,2) );
            gradientMagnitude.at<float>(i, j) = magnitude;
        }
    }

    return gradientMagnitude;
}

void test_gradient(string filename) {
    cv::namedWindow("Original Image", cv::WINDOW_NORMAL);
    cv::namedWindow("Sobel X", cv::WINDOW_NORMAL);
    cv::namedWindow("Sobel Y", cv::WINDOW_NORMAL);
    cv::namedWindow("Gradient Magnitude", cv::WINDOW_NORMAL);

    cv::Mat input = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    if (input.empty()) {
        std::cerr << "Unable to read the image." << std::endl;
        return ;
    }

    cv::Mat sobelX = sobelFilter(input, false);
    cv::Mat sobelY = sobelFilter(input, true);

    cv::Mat gradientMagnitude = calculateGradient(sobelX, sobelY);
    gradientMagnitude.convertTo(gradientMagnitude, CV_8U);

    cv::imshow("Original Image", input);
    cv::imshow("Sobel X", sobelX);
    cv::imshow("Sobel Y", sobelY);
    cv::imshow("Gradient Magnitude", gradientMagnitude);

    cv::waitKey(0);
}

/* 
###################################################################
*/


int t = 20;

void onTrackbarChangeThreshold(int, void*) {
    // Cette fonction sera appelée à chaque changement de la position du slider
    // Elle met à jour la variable threshold
    t = getTrackbarPos("Treshold track", "Output");
}

bool isSignChanging(cv::Mat input, int row, int col)
{
  int pixel1 = input.at<uchar>(row - 1, col - 1);
  int pixel2 = input.at<uchar>(row - 1, col);
  int pixel3 = input.at<uchar>(row - 1, col + 1);
  int pixel4 = input.at<uchar>(row, col - 1);
  int pixel5 = input.at<uchar>(row, col + 1);
  int pixel6 = input.at<uchar>(row + 1, col - 1);
  int pixel7 = input.at<uchar>(row + 1, col);
  int pixel8 = input.at<uchar>(row + 1, col + 1);
  return (pixel1 < 0 && pixel2 < 0 && pixel3 < 0 && pixel4 < 0 && pixel5 < 0 && pixel6 < 0 && pixel7 < 0 && pixel8 < 0) ||
         (pixel1 > 0 && pixel2 > 0 && pixel3 > 0 && pixel4 > 0 && pixel5 > 0 && pixel6 > 0 && pixel7 > 0 && pixel8 > 0);
}

void marrHildrethEdgeDetection(String filename)
{
    namedWindow("Output", WINDOW_NORMAL);
    createTrackbar("Treshold track", "Output", &t, 200, onTrackbarChangeThreshold);
    onTrackbarChangeThreshold(t, nullptr); // Initialisation de la valeur threshold

    cv::Mat input = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    cv::Mat output = cv::Mat::zeros(input.rows, input.cols, CV_8UC1);
    int rowSize = input.rows;
    int columnSize = input.cols;
    
    std::vector <std::vector<double>> Lx = {
        {1.0/4.0, 0.0, -1.0/4.0},
        {2.0/4.0, 0.0, -2.0/4.0},
        {1.0/4.0, 0.0, -1.0/4.0}
    };
    std::vector <std::vector<double>> Ly = {
        {1.0/4.0, 2.0/4.0, 1.0/4.0},
        {0.0, 0.0, 0.0},
        {-1.0/4.0, -2.0/4.0, -1.0/4.0}
    };

    for(int row = 1; row < rowSize - 1; row++) {
        for(int col = 1; col < columnSize - 1; col++) {

            double pixel = input.at<uchar>(row, col);
            double pixel1 = input.at<uchar>(row - 1, col - 1);
            double pixel2 = input.at<uchar>(row - 1, col);
            double pixel3 = input.at<uchar>(row - 1, col + 1);
            double pixel4 = input.at<uchar>(row, col - 1);
            double pixel5 = input.at<uchar>(row, col + 1);
            double pixel6 = input.at<uchar>(row + 1, col - 1);
            double pixel7 = input.at<uchar>(row + 1, col);
            double pixel8 = input.at<uchar>(row + 1, col + 1);

            //apply laplacian to the pixel 
            double convolutedPixelX =
            pixel1 * Lx[0][0] +
            pixel2 * Lx[0][1] +
            pixel3 * Lx[0][2] +
            pixel4 * Lx[1][0] +
            pixel  * Lx[1][1] +
            pixel5 * Lx[1][2] +
            pixel6 * Lx[2][0] +
            pixel7 * Lx[2][1] +
            pixel8 * Lx[2][2];

            double convolutedPixelY =
            pixel1 * Ly[0][0] +
            pixel2 * Ly[0][1] +
            pixel3 * Ly[0][2] +
            pixel4 * Ly[1][0] +
            pixel  * Ly[1][1] +
            pixel5 * Ly[1][2] +
            pixel6 * Ly[2][0] +
            pixel7 * Ly[2][1] +
            pixel8 * Ly[2][2];


            double convolutedPixel = sqrt(pow(convolutedPixelX, 2) + pow(convolutedPixelY, 2));
            
            if (isSignChanging(input, row, col) && convolutedPixel >= t) {
                convolutedPixel -= 255;
                output.at<uchar>(row, col) = convolutedPixel;
            } else {
                output.at<uchar>(row, col) = 255;
            }
        }
    }

    imshow("Output", output);
    waitKey(0);
}


/* 
###################################################################
*/



int main(int argc, char* argv[]) {
    //marrHildrethEdgeDetection(argv[1]);
    test_rehausse_contraste(argv[1]);
    return 1;
}


