template<typename T>
T generate_random_values(T lower_bound, T upper_bound)
{
    static std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
    if constexpr (std::is_floating_point<T>::value)
    {
        std::uniform_real_distribution<T> dis(lower_bound, upper_bound);
        return dis(gen);
    }
    else if constexpr (std::is_integral<T>::value)
    {
        std::uniform_int_distribution<T> dis(lower_bound, upper_bound);
        return dis(gen);
    }

    return T(); // Fallback (though this should never be hit)
}


template<typename T>
std::vector<std::vector<T>> create_matrix(int M, int N, bool is_filled)
{
    std::vector<std::vector<T>> matrix(M, std::vector<T>(N,0));
    if (!is_filled) return matrix;
    for(size_t i = 0; i < M; i++)
    {
        for(size_t j =0; j < N; j++)
        {
         if constexpr (std::is_floating_point<T>::value)
            {
                matrix[i][j] = generate_random_values<T>(0.1, 1.0);
                continue;
            }
        matrix[i][j] = generate_random_values<T>(1, 100);
    }
    }
  
    return matrix;

}

template<typename T>
void print_matrix(std::vector<std::vector<T>> &matrix){

        for(size_t i = 0; i < matrix.size(); i++)
        {
            for (size_t j = 0; j < matrix[0].size(); j++)
            {
                std::cout <<matrix[i][j];
                if(j != matrix[0].size() - 1) std::cout<< ", ";
            }
            if(i != matrix.size() - 1) std::cout << std::endl;
        }
                
            std::cout<<std::endl;
            std::cout<<std::endl;

}


template<typename T>
std::vector<T> flatten_matrix(std::vector<std::vector<T>> &input_matrix)
{
    std::vector<T> res;

    for(auto &row: input_matrix)
    {
        for(auto &elem : row)
        {
            res.push_back(elem);
        }
    }
    return res;
}
template<typename T>
std::vector<std::vector<T>> flat_array_to_matrix(std::vector<T> &flat_array, int M, int N)
{

    std ::vector<std:vector<T>> res_matrix (M, std::vector<T>(N,0));
    for(size_t i = 0; i < M; i++ )
    {
        for (size_t j = 0; j < N; j++)
        {
            res_matrix[i][j] = flat[ i*N + j];
        }
    }

    return res_matrix;

}

template<typename T>
bool is_a_match(std::vector<std::vector<T>> &cpu_matrix, 
                std::vector<std::vector<T>> &gpu_matrix, 
                T tolerance) {
    if (cpu_matrix.size() != gpu_matrix.size() || cpu_matrix[0].size() != gpu_matrix[0].size()) {
        std::cout<<"non trivial difference detected " <<std::endl;
        return false;
    }

    for (size_t i = 0; i < cpu_matrix.size(); ++i) {
        for (size_t j = 0; j < cpu_matrix[i].size(); ++j) {
            if (std::fabs(cpu_matrix[i][j] - gpu_matrix[i][j]) > tolerance) {
                std::cout<<"non trivial difference detected " <<std::endl;
                return false;
            }
        }
    }

    std::cout<<"Everything checks out" <<std::endl;
    return true;
}

template <typename T>
bool matrix_is_empty(std::vector<std::vector<T>> &matrix)
{
    return matrix.size() == 0;
}

template <typename T>
bool matrix_addition_is_possible(std::vector<std::vector<T>> &matrix_A,
 std::vector<std::vector<T>> &matrix_B)
{
    return matrix_A.size() == matrix_B.size() && matrix_A[0].size() == matrix_B[0].size();
}

template <typename T>
bool matrix_multiplication_is_possible(std::vector<std::vector<T>> &matrix_A, 
std::vector<std::vector<T>> &matrix_B)
{
    return matrix_A[0].size() == matrix_B.size();
}