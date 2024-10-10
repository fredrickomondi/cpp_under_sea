


template<typename T>
void gpu_multiply_matrix_naive(std::vector<std::vector<T>> &matrix_A, std::vector<std::vector<T>> &matrix_B, std::vector<std::vector<T>> &matrix_C, sycl::queue q) {

    std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << std::endl;

    auto flat_matrix_a = flatten_matrix(matrix_A);
    auto flat_matrix_b = flatten_matrix(matrix_B);
    auto flat_matrix_c = flatten_matrix(matrix_C);
    size_t M = matrix_A.size();
    size_t N = matrix_A[0].size();

    buffer<T, 1> buffer_A(flat_matrix_a.data(), range<1>(M * N));
    buffer<T, 1> buffer_B(flat_matrix_b.data(), range<1>(M * N));
    buffer<T, 1> buffer_C(flat_matrix_c.data(), range<1>(M * N));

    auto start = std::chrono::high_resolution_clock::now();
    q.submit([&](handler &h) {

        auto a = buffer_A.template get_access<access::mode::read>(h);
        auto b = buffer_B.template get_access<access::mode::read>(h);
        auto c = buffer_C.template get_access<access::mode::write>(h);

        h.parallel_for(range<2>(M, N), [=](id<2> idx) {
            size_t j = idx[0];
            size_t i = idx[1];
            for (size_t k = 0; k < N; ++k) {
                c[j * N + i] += a[j * N + k] * b[k * N + i];
            }
        });

    });

    q.wait();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_time = (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000.0;
    std::cout << "Elapsed time for naive GPU matrix multiplication is: " << elapsed_time << " seconds" << std::endl;

// Read data back to host
    auto host_result_access = buffer_C.template get_host_access();
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix_C[i][j] = host_result_access[i * N + j];  // Copy data back
        }
    }
}


template<typename T>
void gpu_multiply_matrix_optimized(std::vector<std::vector<T>> &matrix_A, std::vector<std::vector<T>> &matrix_B, std::vector<std::vector<T>> &matrix_C, sycl::queue q) {

    std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << std::endl;

    auto flat_matrix_a = flatten_matrix(matrix_A);
    auto flat_matrix_b = flatten_matrix(matrix_B);
    auto flat_matrix_c = flatten_matrix(matrix_C);
    size_t M = matrix_A.size();
    size_t N = matrix_A[0].size();

    buffer<T, 1> buffer_A(flat_matrix_a.data(), range<1>(M * N));
    buffer<T, 1> buffer_B(flat_matrix_b.data(), range<1>(M * N));
    buffer<T, 1> buffer_C(flat_matrix_c.data(), range<1>(M * N));

    auto start = std::chrono::high_resolution_clock::now();
    q.submit([&](handler &h) {

        auto a = buffer_A.template get_access<access::mode::read>(h);
        auto b = buffer_B.template get_access<access::mode::read>(h);
        auto c = buffer_C.template get_access<access::mode::write>(h);

        h.parallel_for(range<2>(M, N), [=](id<2> idx) {
            size_t j = idx[0];
            size_t i = idx[1];
            size_t c_index = j * N + i;
            size_t a_index_base = j * N;
            T res = 0;
            for (size_t k = 0; k < N; ++k) {
                res += a[a_index_base + k] * b[k * N + i];  // change 1 use a local variable in private memory
            }
           
            c[c_index] = res;
        });

    });

    q.wait();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_time = (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000.0;
    std::cout << "Elapsed time for optimized GPU matrix multiplication is: " << elapsed_time << " seconds" << std::endl;

// Read data back to host
    auto host_result_access = buffer_C.template get_host_access();
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix_C[i][j] = host_result_access[i * N + j];  // Copy data back
        }
    }
}

template <typename T>
void gpu_add_matrix(std::vector<std::vector<T>>&matrix_A, std::vector<std::vector<T>> &matrix_B, std::vector<std::vector<T>>&matrix_C, sycl::queue q)
{
    if(!matrix_is_empty(matrix_A) && !matrix_is_empty(matrix_B) && matrix_addition_is_possible(matrix_A,matrix_B))
    {
        std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << std::endl;
        std::vector<T> flat_matrix_A = flatten_matrix(matrix_A);
        std::vector<T> flat_matrix_B = flatten_matrix(matrix_B);
        std::vector<T> flat_matrix_C = flatten_matrix(matrix_C);
        size_t M = matrix_A.size();
        size_t N = matrix_A[0].size();
        
        buffer<T,1> buffer_A(flat_matrix_A.data(),range<1>(M*N));
        buffer<T,1> buffer_B(flat_matrix_B.data(), range<1>(M*N));
        buffer<T,1> buffer_C(flat_matrix_C.data(),range<1>(M*N));
       

        auto start = std::chrono::high_resolution_clock::now();
        
        q.submit([&] (handler &h ) {
            auto a = buffer_A.template get_access<access::mode::read>(h);
            auto b = buffer_B.template get_access<access::mode::read>(h);
            auto c = buffer_C.template get_access<access::mode::write>(h);
                    
            h.parallel_for(range<1>(M*N), [=] (id<1> idx) {
              
                c[idx] = a[idx] + b[idx];
               
            });
        });
       
        q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/1000.0;
         std::cout << "Elapsed time for GPU matrix addition is: " << elapsed_time << " seconds" << std::endl;
         auto host_result_access = buffer_C.template get_host_access();
          std::cout<<"here 3"<<std::endl;
         for(size_t i = 0; i < M; i++)
         {
            for(size_t j = 0; j < N; j++)
            {
                matrix_C[i][j] = host_result_access[i*N + j];
            }
         }
    }else {
          std::cout<<"ERROR:  operation aborted! Ensure input matrix are non-empty and their dimensions match" << std::endl;

    }
}

