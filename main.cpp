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

/* void onTrackbarChange(int, void*) {
    // Cette fonction sera appelée à chaque changement de la position du slider
    // Elle met à jour la variable alpha
    alpha = getTrackbarPos("alpha (en %)", "Image");
} */

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


cv::Mat gradient(cv::Mat input){
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
        output.at<uchar>(row, col) = convolutedPixel;
      }
  }

  return output;
}



/* cv::Mat calculateGradient(const cv::Mat& Ix, const cv::Mat& Iy) {
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
} */

/* void test_gradient(string filename) {
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
} */

/* 
###################################################################
*/


int t = 20;

void onTrackbarChangeThreshold(int val, void* data) {
  int* ptr_data = (int*)data;
  *ptr_data = val;
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

cv::Mat marrHildreth(cv::Mat input, double sigma){
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
        if(isSignChanging(input, row, col) && convolutedPixel >= sigma){
            convolutedPixel -= 255;
            output.at<uchar>(row, col) = convolutedPixel;
          }else{
            output.at<uchar>(row, col) = 255;
          }
    }
  }
  return output;
}

cv::Mat esquisse(Mat input, int seuilMarr, int proportionT, int longueur)
{
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
            
            if(isSignChanging(input, row, col) && convolutedPixel >= seuilMarr){
                double r = (double)rand() / RAND_MAX;
                if(r < proportionT/100.0) {
                    double angle = atan2(- convolutedPixelX,convolutedPixelY) + M_PI/2 + 0.02*(r-0.5);
                    double l2 = convolutedPixel/255.0 * longueur/10.0;
                    cv::line(
                        output,
                        cv::Point(
                            col + l2 * cos(angle),
                            row + l2 * sin(angle)
                        ),
                        cv::Point(
                            col - l2 * cos(angle),
                            row - l2 * sin(angle)
                        ),
                        cv::Scalar(0, 0, 0)
                    );
                }
            }else{
                output.at<uchar>(row, col) = 255;
            }
        }
    }

    return output;
}


void on_trackbar(int val, void* data) {
  int* ptr_data = (int*)data;
  *ptr_data = val;
}

int test(String filename)
{
    namedWindow( "Youpi");
    int threshold = 20;
    cv::Mat input = cv::imread(filename);
    
    while ( true ) {
        if ( input.channels() == 3 ) {
            cv::cvtColor( input, input, COLOR_BGR2GRAY );
        }
        int keycode = waitKey( 50 );
        int asciicode = keycode & 0xff;
        
        createTrackbar( "threshold (en %)", "Youpi", &threshold, 100,  NULL);
        setTrackbarPos( "threshold (en %)", "Filter", threshold ); // init à 20

        // récupère la valeur courante de threshold
        threshold = getTrackbarPos( "threshold (en %)", "Youpi" );
        if ( asciicode == 'q' ) break;
        if ( asciicode == 'h' ) {
            input = marrHildreth(input, threshold);
        }
        if ( asciicode == 'e' ) {
            input = esquisse(input, threshold, 20, 20);
        }
        if ( asciicode == 'r' ) {
            input = cv::imread(filename);
            cv::cvtColor( input, input, COLOR_BGR2GRAY );
        }
        imshow( "Youpi", input );
    }
    return 0;
}

/* 
###################################################################
*/


int test_video()
{
    namedWindow( "Youpi");               // crée une fenêtre
    VideoCapture cap(0);
    if(!cap.isOpened()) return -1;
    Mat input;
    int alpha = 20;
    int seuilMarr = 20;
    int proportionT = 20;
    int longueur = 20;
    int asciicode = 255;
    int cpt = 0;
    while ( true ) {
        cap >> input;
        if ( input.channels() == 3 ) {
            cv::cvtColor( input, input, COLOR_BGR2GRAY );
        }
        int keycode = waitKey( 50 );
        if(keycode != -1) {
            asciicode = keycode & 0xff;
            if(asciicode == 'a' || asciicode == 'm' || asciicode == 's') cpt++;
        }
        
        createTrackbar( "alpha (en %)", "Youpi", 0, 200,  on_trackbar, &alpha);
        setTrackbarPos( "alpha (en %)", "Youpi", alpha ); // init à 20

        createTrackbar( "seuil Marr", "Youpi", 0, 200,  on_trackbar, &seuilMarr);
        setTrackbarPos( "seuil Marr", "Youpi", seuilMarr ); // init à 20

        createTrackbar( "t (% chance segment)", "Youpi", 0, 100.0, on_trackbar, &proportionT);
        setTrackbarPos( "t (% chance segment)", "Youpi", proportionT ); // init à 20

        createTrackbar( "L longueur trait", "Youpi", 0, 1000, on_trackbar, &longueur);
        setTrackbarPos( "L longueur trait", "Youpi", longueur ); // init à 20

        // récupère la valeur courante de alpha
        alpha = getTrackbarPos( "alpha (en %)", "Youpi" );
        if ( asciicode == 'q' ) break;
        //if a is pressed apply filterM to the image
        if ( asciicode == 'a' ) {
            for (int i = 0; i < cpt; i++)
            {
                Mat M = (Mat_<float>(3, 3) << 1.0 / 16, 2.0 / 16, 1.0 / 16,
                                        2.0 / 16, 4.0 / 16, 2.0 / 16,
                                        1.0 / 16, 2.0 / 16, 1.0 / 16);
                input = filtreM(input, M);
            }
        }
        if ( asciicode == 'm' ) {
            for (int i = 0; i < cpt; i++)
            {
                input = filtreMedianBlur(input);
            }
        }
        if ( asciicode == 's' ) {
            for (int i = 0; i < cpt; i++)
            {
                input = rehaussementContraste(input, alpha);
            }
        }
        if ( asciicode == 'x' ) {
            input = sobelFilter(input, false);
        }
        if ( asciicode == 'y' ) {
            input = sobelFilter(input, true);
        }
        if ( asciicode == 'g' ) {
            input = gradient(input);
        }
        if ( asciicode == 'h' ) {
            input = marrHildreth(input, seuilMarr);
        }
        if ( asciicode == 'e' ) {
            input = esquisse(input, seuilMarr, proportionT, longueur);
        }
        if ( asciicode == 'r' ) {
            cpt = 0;
            asciicode = 255;
        }
        imshow( "Youpi", input );          // l'affiche dans la fenêtreq
    }
    return 0;
}


int main(int argc, char* argv[])
{
    return test_video();
}