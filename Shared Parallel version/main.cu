#include <string>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <sys/time.h>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime_api.h>
#include "utils.h"


using namespace std;
using namespace cv;

__device__ float Eu_distance(float B, float G, float R, float *centroid, int k)
{
    float sum;
    float dist;
    sum = float(pow(B-centroid[k+0], 2) + pow(G-centroid[k+1], 2) + pow(R-centroid[k+2], 2));
    dist = float(sqrt(sum));
    return dist;
}


__global__ void match_point(int *cluster, float *B_c, float *G_c, float *R_c, float *centr, int size_image, int n_threads, int K)
{
	int i = threadIdx.x;

	int size_per_thread = size_image/n_threads;
    int start = i*size_per_thread;
	int end = start + size_per_thread;

	float distance, distance_old;

	if (i >=size_image){ return; }

	if (i==n_threads-1)
    {
        start = (n_threads-1)*size_per_thread;
        end = size_image;
	}

	for(int j = start; j<end; j++)
	{
		for(int k=0; k<K; k++)
		{
			distance = Eu_distance(B_c[j], G_c[j], R_c[j], centr, k*3);

            if (k == 0)
            {
                distance_old = distance;
                cluster[j] = k;
                
            }
            else if (distance<=distance_old)
            {
                distance_old = distance;
                cluster[j] = k;
            }

		}
	}
}

__global__ void update_cluster(int *cluster, float *centroid, float *B_c, float *G_c, float *R_c, int size_image, int n_threads, int K)
{

	extern __shared__ float sdata[];
	float *nValue = sdata;
	float *Bdata = &nValue[(K)*n_threads-1];
	float *Gdata = &Bdata[(K)*n_threads];
	float *Rdata = &Gdata[(K)*n_threads];

	unsigned int tid = threadIdx.x;
	int k = blockIdx.x;

	int size_per_thread = int(size_image/n_threads);
    int start = tid*size_per_thread;
	int end = start + size_per_thread;

	float count = 0;
	float B = 0;
	float G = 0;
	float R = 0;	

	if (tid >=size_image){ return; }

	if (tid==n_threads-1)
    {
        start = (n_threads-1)*size_per_thread;
        end = size_image;
	}
	for(int j = start; j < end; j++)
	{
		if(cluster[j] == k)
		{
            B = B + (B_c[j]);
            G = G + (G_c[j]);
            R = R + (R_c[j]);
            count = count + 1; 
		}
	}

    nValue[tid] = count;
    Bdata[tid] = B;
    Gdata[tid] = G;
    Rdata[tid] = R;

    __syncthreads();

    for(unsigned int s=1; s < blockDim.x; s *= 2) 
	{
		if(tid % (2*s) == 0)
		{
			nValue[tid] += nValue[tid + s];
			Bdata[tid] += Bdata[tid + s];
			Gdata[tid] += Gdata[tid + s];
			Rdata[tid] += Rdata[tid + s];
		}
		__syncthreads();
	}


	if(tid == 0)
	{
		if (nValue[0] != 0)
		{
			centroid[k*3 + 0] = Bdata[0] / nValue[0];
			centroid[k*3 + 1] = Gdata[0] / nValue[0];
			centroid[k*3 + 2] = Rdata[0] / nValue[0];
		}
	}
}


int main(int argc, char**argv)
{

	clock_t start, end;
	double cpu_time_used;


	char *str_image;
    str_image = argv[1];
    cout<< str_image << endl;

    cv::Mat image;	
    image = cv::imread(str_image , CV_LOAD_IMAGE_COLOR);

    int size_image = (image.rows*image.cols);
    cout << size_image << endl;

    if(! image.data ) {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    unsigned int k;
    cout << "Selezionare il numero di Cluster desiderati:"<< endl;
    cin>>k;

    unsigned int n_threads;
    cout << "Selezionare il numero di thread desiderati:"<< endl;
    cin>>n_threads;


    start = clock();

	float *B_channel =  new float [size_image];
	float *G_channel =  new float [size_image];
	float *R_channel =  new float [size_image];

	float *B_c, *G_c, *R_c;

	// vettorizzazione delle coordinate BGR per portarle in __device__
	for(int i=0; i<size_image; i++)
	{
 	   B_channel[i] = float(image.data[image.channels()*i+0]);
       G_channel[i] = float(image.data[image.channels()*i+1]);
       R_channel[i] = float(image.data[image.channels()*i+2]);
	}

	cudaMalloc(&B_c, size_image* sizeof(float));
	cudaMalloc(&G_c, size_image* sizeof(float));
	cudaMalloc(&R_c, size_image* sizeof(float));
	cudaMemcpy(B_c, B_channel, size_image* sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(G_c, G_channel, size_image* sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(R_c, R_channel, size_image* sizeof(float), cudaMemcpyHostToDevice);

    // Inizializzazione dei cluster
    float *Centroids = new float[k*3];
    float *centr;
    // Coordinate dei centroidi
    float C_x, C_y, C_z;

    int count = 0;
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

    // verifica della posizione dei centroidi
    for(int j=0; j < k*3; j++){
        cout << Centroids[j] << " ";
        if (j%3 == 2){
            cout << " Centroide numero: " << count << "\n";
            count ++;
        }
    }

    cudaMalloc(&centr, k*3*sizeof(float));
    cudaMemcpy(centr, Centroids, k*3*sizeof(float), cudaMemcpyHostToDevice);

    // variabile utile per il confronto dei vecchi centroidi con i nuovi
    float *Centroids_old = new float [k*3];
    // variabile che indica l'appartenenza a quale cluster
    int *Clusters = new int [size_image];
    int *clust;
    cudaMalloc(&clust, size_image*sizeof(int));
    // variabile di errore tra il vecchio e il nuoce centroide;
    float error = 100000;

    while (error > 0){

    	match_point<<<1,n_threads>>>(clust, B_c, G_c, R_c, centr, size_image, n_threads, k);
		cudaThreadSynchronize();

		cudaMemcpy(Clusters, clust, size_image*sizeof(int), cudaMemcpyDeviceToHost);

		// Deep copy
        for(int i=0; i < k*3; i++){
            Centroids_old[i] = Centroids[i];
        }

        // calcolo dei nuovi centroidi
        update_cluster<<<k,n_threads, (k)*n_threads*sizeof(float)+(k)*n_threads*sizeof(float)+(k)*n_threads*sizeof(float)+(k)*n_threads*sizeof(float)>>>(clust, centr, B_c, G_c, R_c, size_image, n_threads, k);
        cudaThreadSynchronize();
        cudaMemcpy(Centroids, centr, k*3*sizeof(float), cudaMemcpyDeviceToHost);

        for (int j=0; j < k*3; j = j + 3)
        {
            cout << "Centroidi con metodo parallelo: " << Centroids[j+0] << " " <<Centroids[j+1] << " " <<Centroids[j+2] << endl;
        }

        error = error_distance(Centroids, Centroids_old, k);
        cout << error << endl;
        //cin>>count;
    }

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    cout << "Tempo di esecuzione: " << cpu_time_used << endl;

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
    imshow("Display window", image);                 // Show our image inside it.

    waitKey(0);
    return 0;

}

