#include <iostream>
#include <stdio.h>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

    char *str_image;
    str_image = argv[1];
    cout<< str_image << endl;

    cv::Mat image;
    image = cv::imread(str_image , CV_LOAD_IMAGE_COLOR);

    if(! image.data ) {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    int size_image = (image.rows*image.cols);

    clock_t start, end;
    double cpu_time_used;
    
    // Numero di cluster
    unsigned int k;
    cout << "Selezionare il numero di Cluster desiderati:"<< endl;
    cin>>k;

    start = clock();


    // Inizializzazione dei cluster
    float *Centroids = new float[k*3];
    // Coordinate dei centroidi
    float C_x, C_y, C_z;
    // seed per la funzione rand
    // Commentare se si vuole avere sempre gli stessi centroidi
    //srand(time(NULL));
    for (int j=0; j < k*3; j = j + 3){
        C_x = abs(rand()%255);
        C_y = abs(rand()%255);
        C_z = abs(rand()%255);
        Centroids[j+0] = C_x;
        Centroids[j+1] = C_y;
        Centroids[j+2] = C_z;
    }

    int count = 0;

    // verifica della posizione dei centroidi
    for(int j=0; j < k*3; j++){
        cout << Centroids[j] << " ";
        if (j%3 == 2){
            cout << " Cluster numero: " << count << "\n";
            count ++;
        }
    }

    // variabile utile per il confronto dei vecchi centroidi con i nuovi
    float *Centroids_old = new float [k*3];
    // variabile che indica l'appartenenza a quale cluster
    float *Clusters = new float [size_image];
    // variabile di errore tra il vecchio e il nuoce centroide;
    float error = 100000;
    // semplici variabili
    float distance, distance_old;
    float B, G, R;
    count = 0;
    int n = 0;
    int r;


    while (error > 0){

        //assegnamento dei cluster ai punti, rispetto ai centroidi più vicini
        for (int i=0; i<size_image; i++)
        {
            B = float(image.data[image.channels()*i+0]);
            G = float(image.data[image.channels()*i+1]);
            R = float(image.data[image.channels()*i+2]);

            for(int j=0; j<k; j++)
            {
                distance = E_distance(B, G, R, Centroids, j*3);

                if (j == 0)
                {
                    distance_old = distance;
                    Clusters[i] = j;
                }
                else if (distance<=distance_old)
                {
                    distance_old = distance;
                    Clusters[i] = j;
                }
            }
        }

        // Deep copy
        for(int i=0; i < k*3; i++){
            Centroids_old[i] = Centroids[i];
        }

        // calcolo dei nuovi centroidis
        n = 0;
        for(int i=0; i < k*3; i=i+3){
            B=0;
            G=0;
            R=0;
            count=0;
            for(int j=0; j < size_image; j++)
            {
                if (Clusters[j] == n)
                {
                    B = B + float(image.data[image.channels()*j+0]);
                    G = G + float(image.data[image.channels()*j+1]);
                    R = R + float(image.data[image.channels()*j+2]);
                    count = count + 1;
                }

            }

            if(count!=0)
            {
                Centroids[i + 0] = (B) / count;
                Centroids[i + 1] = (G) / count;
                Centroids[i + 2] = (R) / count;
            }
            n++;
        }
        for (int j=0; j < k*3; j = j + 3)
        {
            cout << Centroids[j+0] << " " <<Centroids[j+1] << " " <<Centroids[j+2] << endl;
        }
        error = error_distance(Centroids, Centroids_old, k);

       cout << "errore: " << error << endl;
    }

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    cout << "Tempo di esecuzione: " << cpu_time_used << endl;

    // fuori dal ciclio while, serve solo a stampare la roba e essere sicuri
    for(int i=0; i<k; i++)
    {
        for (int j = 0; j < size_image; j++) {
            if (Clusters[j] == i){
                image.data[image.channels()*j+0] = uchar(Centroids[i+0]);
                image.data[image.channels()*j+1] = uchar(Centroids[i+1]);
                image.data[image.channels()*j+2] = uchar(Centroids[i+2]);
            }
        }
        cout<<"it's working" << endl;
    }

    resize(image, image, Size(1024, 768), 0, 0, INTER_CUBIC);

    namedWindow( "Display window", WINDOW_AUTOSIZE );    // Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.

    waitKey(0);
    return 0;
}