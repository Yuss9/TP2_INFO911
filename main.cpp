#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;


Mat filtreM(Mat input) {
    // Définition du masque M (filtre moyenneur)
    Mat M = (Mat_<float>(3, 3) << 1.0 / 16, 2.0 / 16, 1.0 / 16,
                                        2.0 / 16, 4.0 / 16, 2.0 / 16,
                                        1.0 / 16, 2.0 / 16, 1.0 / 16);

    // Convolution de l'image avec le filtre moyenneur
    Mat result;
    filter2D(input, result, -1, M, Point(-1, -1), 0, BORDER_DEFAULT);

    return result;
}

int main(int argc, char* argv[]) {
    namedWindow("Youpi"); // crée une fenêtre
    Mat input = imread(argv[1]); // lit l'image donnée en paramètre

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
                input = filtreM(input);
            }
            imshow("Youpi", input); // l'affiche dans la fenêtre
        } else {
            imshow("Youpi", input); // l'affiche dans la fenêtre
        }
    }

    imwrite("result.png", input); // sauvegarde le résultat

    return 0;
}
