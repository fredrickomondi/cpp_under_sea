template <typename T>
void cpu_add_matrix(std::vector<std::vector<T>>&matrix_A, std::vector<std::vector<T>>&matrix_B, std::vector<std::vector<T>>&matrix_C )
{

      if(!matrix_is_empty(matrix_A) && !matrix_is_empty(matrix_B) && matrix_addition_is_possible(matrix_A,matrix_B))
      {
       
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i <matrix_A.size(); i++)
        {
          for(size_t j = 0; j < matrix_A[0].size(); j++)
          {
            matrix_C[i][j] = matrix_A[i][j] + matrix_B[i][j];
          }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_time = (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() ) /1000.0;
    
        std::cout<<"Elapsed time for CPU matrix addition is : "<<elapsed_time<< " seconds" << std::endl;
        
      }else{
        std::cout<<"ERROR:  operation aborted! Ensure matrix are non-empty and their dimensions match" <<std::endl;
        
      }
      
  return;
}



template<typename T>
void cpu_multiply_matrix(std::vector<std::vector<T>> &matrix_A, std::vector<std::vector<T>>&matrix_B, std::vector<std::vector<T>>&matrix_C)
{

    // by definition, columns of matrix A must be equal to the rows of matrix B , this means the result matrix will be of dimensions rowsof
    // matrix A by columns of matrix B

    if(matrix_A[0].size() != matrix_B.size())
    {
        std::cout<<"ERROR: matrix_A * matrix_B is undefined because cols(mamtrix_A) is not equal to rows(matrix_B)";
        return;
    }


    size_t rows_A = matrix_A.size();
    size_t rows_B = matrix_B.size();
    size_t cols_B = matrix_B[0].size();

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t  i = 0; i < rows_A; i++)
    {
        for(size_t j = 0; j< cols_B; j++)

        {
            for(size_t k = 0; k < rows_B; k++)
            {
                matrix_C[i][j] += matrix_A[i][k] * matrix_B[k][j];
            }
        }
    }
   auto end = std::chrono::high_resolution_clock::now();
   auto elapsed_time = (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() ) /1000.0;
    
    std::cout<<"Elapsed time for CPU matrix multiplication is : "<<elapsed_time<< " seconds" << std::endl;

}
