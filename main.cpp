#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    namedWindow("TP2");               // crée une fenêtre
    Mat input = imread(argv[1]);     // lit l'image donnée en paramètre
    
    if (input.channels() == 3)
        cv::cvtColor(input, input, COLOR_BGR2GRAY);
    
    while (true) {
        int keycode = waitKey(50);
        int asciicode = keycode & 0xff;
        if (asciicode == 'q') break;
        imshow("TP2", input);            // l'affiche dans la fenêtre
    }

    imwrite("result.png", input);          // sauvegarde le résultat
}  