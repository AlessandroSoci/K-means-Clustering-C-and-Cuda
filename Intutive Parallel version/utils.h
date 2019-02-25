//
// Created by alessandro on 02/01/19.
//

#ifndef K_MEANS_UTILS_H
#define K_MEANS_UTILS_H


#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>

using namespace std;

float E_distance(float B, float G, float R, float *centroid, int k){
    float sum;
    float dist;
    sum = float(pow(B-centroid[k+0], 2) + pow(G-centroid[k+1], 2) + pow(R-centroid[k+2], 2));
    dist = u_int((sqrt(sum)));
    //cout << dist << endl;
    return dist;
}

float error_distance(float *centroid, float *old_centroid, int K){
    float error=0;
    float dist;
    for(int i=0; i < K*3; i=i+3)
    {
        dist = float(sqrt(pow(old_centroid[i+0]-centroid[i+0],2) + pow(old_centroid[i+1]-centroid[i+1], 2) + pow(old_centroid[i+2]-centroid[i+2], 2)));
        error = error + dist;
//        cout << u_int(centroid[i+0]) << " " << u_int(centroid[i+1]) << " " << u_int(centroid[i+2]) << endl;
//        cout << u_int(old_centroid[i+0]) << " " << u_int(old_centroid[i+1]) << " " << u_int(old_centroid[i+2]) << endl;
//        cout << "ERRORE in error_distance: " << error << endl;
    }

    return error;
}
#endif //K_MEANS_UTILS_H
