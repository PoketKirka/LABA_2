#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include <string.h>
#include <cblas.h>

#define N 2048
#define NUM_TESTS 1
#define SAMPLE_SIZE 1000  
#define TOLERANCE 1e-3f

void generate_matrix(float complex *matrix) {
    for (int i = 0; i < N * N; i++) {
        float real = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        float imag = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        matrix[i] = real + imag * I;
    }
}

void naive_matrix_mult(const float complex *A, const float complex *B, float complex *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void optimized_matrix_mult(const float complex *A, const float complex *B, float complex *C) {
    const int block_size = 32; 
    memset(C, 0, N * N * sizeof(float complex));
    
    for (int bi = 0; bi < N; bi += block_size) {
        for (int bj = 0; bj < N; bj += block_size) {
            for (int bk = 0; bk < N; bk += block_size) {
                for (int i = bi; i < bi + block_size && i < N; i++) {
                    for (int j = bj; j < bj + block_size && j < N; j++) {
                        float complex sum = C[i * N + j];
                        for (int k = bk; k < bk + block_size && k < N; k++) {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

int verify_results(const float complex *C1, const float complex *C2) {
    int mismatches = 0;
    const int step = (N * N) / SAMPLE_SIZE;
    
    for (int i = 0; i < N * N; i += step) {
        float diff = cabsf(C1[i] - C2[i]);
        if (diff > TOLERANCE) {
            if (mismatches < 5) { 
                printf("Mismatch at %d: (%.4f + %.4fi) vs (%.4f + %.4fi), diff=%.4f\n", 
                       i, crealf(C1[i]), cimagf(C1[i]), 
                       crealf(C2[i]), cimagf(C2[i]), diff);
            }
            mismatches++;
        }
    }
    
    if (mismatches > 0) {
        printf("Total mismatches: %d out of %d samples checked\n", mismatches, SAMPLE_SIZE);
        return 0;
    }
    return 1;
}

void print_performance(double time, const char *method) {
    double complexity = 2.0 * N * N * N;
    double mflops = complexity / time * 1e-6;
    printf("%s: Time = %.3f sec, Performance = %.2f MFlops\n", method, time, mflops);
}


int main() {
    float complex *A = malloc(N * N * sizeof(float complex));
    float complex *B = malloc(N * N * sizeof(float complex));
    float complex *C_naive = malloc(N * N * sizeof(float complex));
    float complex *C_blas = malloc(N * N * sizeof(float complex));
    float complex *C_optimized = malloc(N * N * sizeof(float complex));
    
    if (!A || !B || !C_naive || !C_blas || !C_optimized) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    srand(time(NULL));
    generate_matrix(A);
    generate_matrix(B);
    
    clock_t start = clock();
    naive_matrix_mult(A, B, C_naive);
    double naive_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    print_performance(naive_time, "Naive method");
    
    start = clock();
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
               N, N, N, 
               &(float complex){1.0f + 0.0f*I}, 
               A, N, 
               B, N, 
               &(float complex){0.0f + 0.0f*I}, 
               C_blas, N);
    double blas_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    print_performance(blas_time, "BLAS method");
    
    printf("\nVerifying Naive vs BLAS...\n");
    if (!verify_results(C_naive, C_blas)) {
        printf("Warning: Naive and BLAS results differ significantly!\n");
    } else {
        printf("Naive and BLAS results match within tolerance.\n");
    }
    
    start = clock();
    optimized_matrix_mult(A, B, C_optimized);
    double optimized_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    print_performance(optimized_time, "Optimized method");
    
    printf("\nVerifying Optimized vs BLAS...\n");
    if (!verify_results(C_blas, C_optimized)) {
        printf("Warning: Optimized and BLAS results differ significantly!\n");
    } else {
        printf("Optimized and BLAS results match within tolerance.\n");
    }
    
    double optimized_perf = (2.0 * N * N * N) / optimized_time * 1e-6;
    double blas_perf = (2.0 * N * N * N) / blas_time * 1e-6;
    double percentage = (optimized_perf / blas_perf) * 100;
    printf("\nOptimized method achieves %.1f%% of BLAS performance\n", percentage);
    
    printf("\nИсаков Андрей Витальевич 090304-РПИа-о24\n");
    
    free(A);
    free(B);
    free(C_naive);
    free(C_blas);
    free(C_optimized);
    
    return 0;
}
