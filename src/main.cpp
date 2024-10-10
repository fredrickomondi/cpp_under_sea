#include <CL/sycl.hpp>
#include <iostream>
#include <random>
#include <ctime>
#include <chrono>
#include "utils.h"
#include "host_operations.h"
#include "device_operations.h"

using namespace sycl;


int main()
{

    //DEFINE MATRIX SIZES - FOR NOW ONLY WORKS WITH SQUARE MATRIXES
    int ROWS = 1024*2;
    int COLS = 1024*2;
    queue q;

    auto matrix_A = create_matrix<float>(ROWS,COLS,true);
    auto matrix_B = create_matrix<float>(ROWS,COLS,true);
    auto cpu_matrix_C = create_matrix<float>(ROWS,COLS,false);
    auto gpu_matrix_C = create_matrix<float>(ROWS,COLS,false);
    auto gpu_matrix_C_optimized = create_matrix<float>(ROWS,COLS,false);

    std::cout<<"Getting started......" <<std::endl;

   /*UNCOMMENT LINE BELOW TO PERFORM CPU MATRIX ADDITION*/
   //cpu_add_matrix<float>(matrix_A,matrix_B, cpu_matrix_C);


    /*UNCOMMENT LINE BELOW TO PERFORM GPU MATRIX ADDITION*/
   // gpu_add_matrix<float>(matrix_A,matrix_B, gpu_matrix_C,q);

   
   
   

    /*UNCOMMENT LINE BELOW TO PERFORM NAIVE GPU MATRIX MULTIPLICATION*/
    //gpu_multiply_matrix_naive<float>(matrix_A,matrix_B,gpu_matrix_C,q);


    /*UNCOMMENT LINE BELOW TO PERFORM CPU MATRIX MULTIPLICATION*/
   // cpu_multiply_matrix <float> (matrix_A,matrix_B,cpu_matrix_C);
    


    /*UNCOMMENT LINE BELOW TO PERFORM NAIVE GPU MATRIX MULTIPLICATION*/
    gpu_multiply_matrix_naive<float>(matrix_A,matrix_B,gpu_matrix_C,q);



    /*UNCOMMENT LINE BELOW TO PERFORM OPTIMIZED GPU MATRIX MULTIPLICATION*/
   gpu_multiply_matrix_optimized<float>(matrix_A, matrix_B, gpu_matrix_C_optimized,q);


//   print_matrix(cpu_matrix_C);
//   print_matrix(gpu_matrix_C);
//   print_matrix(gpu_matrix_C_optimized);
   is_a_match<float>(gpu_matrix_C_optimized,gpu_matrix_C, 1e-4);

    return 0;
}