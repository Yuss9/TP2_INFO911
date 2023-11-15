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


int alpha = 20; // Valeur initiale du coefficient alpha

void onTrackbarChange(int, void*) {
    // Cette fonction sera appelée à chaque changement de la position du slider
    // Elle met à jour la variable alpha
    alpha = getTrackbarPos("alpha (en %)", "Image");
}

// ici on utilise filter2D de opencv mais on peut aussi utiliser notre fonction
Mat rehaussementContraste(Mat input, int alpha) {
    Mat laplacianMask = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5 + alpha / 100.0, -1, 0, -1, 0);
    Mat result;
    filter2D(input, result, -1, laplacianMask, Point(-1, -1), 0, BORDER_DEFAULT);

    return result;
}


void test_rehausse_contraste(String filename){
    namedWindow("Image");
    createTrackbar("alpha (en %)", "Image", &alpha, 200, onTrackbarChange);
    onTrackbarChange(alpha, nullptr); // Initialisation de la valeur alpha

    Mat input = imread(filename);

    while (true) {
        int keycode = waitKey(50);
        int asciicode = keycode & 0xff;

        if (input.channels() == 3)
            cvtColor(input, input, COLOR_BGR2GRAY);


        if (asciicode == 'q')
            break;
        else if (asciicode == 's') {
            // Appliquer le rehaussement de contraste à l'image en utilisant le coefficient alpha
            Mat result = rehaussementContraste(input, alpha);
            imshow("Image", result);
        } else {
            imshow("Image", input);
        }
    }

    imwrite("result.png", input);
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

void marrHildrethEdgeDetection(String filename){
    Mat input = imread(filename, IMREAD_GRAYSCALE);
    int threshold = 90;

    if (input.empty()) {
        cerr << "Error: Couldn't load the image." << endl;
        return;
    }

    namedWindow("Original Image", WINDOW_NORMAL);
    imshow("Original Image", input);

    Mat inputImage = imread(filename);

    Mat grayImage;
    cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);

    Mat gradX, gradY;
    Sobel(grayImage, gradX, CV_32F, 1, 0, 3); // Sobel filter for horizontal gradient
    Sobel(grayImage, gradY, CV_32F, 0, 1, 3); // Sobel filter for vertical gradient

    Mat magnitude, direction;
    cartToPolar(gradX, gradY, magnitude, direction);

    Mat outputImage = Mat::ones(grayImage.size(), CV_8UC1) * 255;

    for (int i = 0; i < grayImage.rows; i++) {
        for (int j = 0; j < grayImage.cols; j++) {
            if (magnitude.at<float>(i, j) >= threshold) {
                outputImage.at<uchar>(i, j) = 0;
            }
        }
    }

    // namedWindow("Sobel Vertical", WINDOW_NORMAL);
    // imshow("Sobel Vertical", gradY);

    // namedWindow("Sobel Horizontal", WINDOW_NORMAL);
    // imshow("Sobel Horizontal", gradX);

    namedWindow("Output", WINDOW_NORMAL);
    imshow("Output", outputImage);
    
    waitKey(0);
}

int main(int argc, char* argv[]) {
    marrHildrethEdgeDetection(argv[1]);
    return 0;
}


