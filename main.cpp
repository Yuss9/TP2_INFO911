#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;

#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
using namespace cv;

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

int main(int argc, char* argv[])
{
    namedWindow("Youpi"); 
    Mat input = imread(argv[1]); 

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
            imshow("Youpi", input);
        } else {
            imshow("Youpi", input);
        }
    }

    imwrite("result.png", input); 

    return 0;
}
