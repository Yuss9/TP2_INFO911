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

int main(int argc, char* argv[]) {
    test_filtre_median(argv[1]);
    return 0;
}