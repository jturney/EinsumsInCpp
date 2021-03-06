#pragma once

#include "einsums/STL.hpp"

#include <complex>
#include <vector>

// Namespace for BLAS and LAPACK routines.
namespace einsums::blas {

// Some of the backends may require additional initialization before their use.
void initialize();
void finalize();

/*!
 * Performs matrix multiplication for general square matices of type double.
 */
void sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta,
           float *c, int ldc);
void dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta,
           double *c, int ldc);
void cgemm(char transa, char transb, int m, int n, int k, std::complex<float> alpha, const std::complex<float> *a, int lda,
           const std::complex<float> *b, int ldb, std::complex<float> beta, std::complex<float> *c, int ldc);
void zgemm(char transa, char transb, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda,
           const std::complex<double> *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc);

template <typename T>
void gemm(char transa, char transb, int m, int n, int k, T alpha, const T *a, int lda, const T *b, int ldb, T beta, T *c, int ldc);

template <>
inline void gemm<float>(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb,
                        float beta, float *c, int ldc) {
    sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<double>(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb,
                         double beta, double *c, int ldc) {
    dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<std::complex<float>>(char transa, char transb, int m, int n, int k, std::complex<float> alpha,
                                      const std::complex<float> *a, int lda, const std::complex<float> *b, int ldb,
                                      std::complex<float> beta, std::complex<float> *c, int ldc) {
    cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<std::complex<double>>(char transa, char transb, int m, int n, int k, std::complex<double> alpha,
                                       const std::complex<double> *a, int lda, const std::complex<double> *b, int ldb,
                                       std::complex<double> beta, std::complex<double> *c, int ldc) {
    zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/*!
 * Performs matrix vector multiplication.
 */
void sgemv(char transa, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y, int incy);
void dgemv(char transa, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy);
void cgemv(char transa, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *x,
           int incx, std::complex<float> beta, std::complex<float> *y, int incy);
void zgemv(char transa, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda, const std::complex<double> *x,
           int incx, std::complex<double> beta, std::complex<double> *y, int incy);

template <typename T>
void gemv(char transa, int m, int n, T alpha, const T *a, int lda, const T *x, int incx, T beta, T *y, int incy);

template <>
inline void gemv<float>(char transa, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y,
                        int incy) {
    sgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<double>(char transa, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta,
                         double *y, int incy) {
    dgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<std::complex<float>>(char transa, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda,
                                      const std::complex<float> *x, int incx, std::complex<float> beta, std::complex<float> *y, int incy) {
    cgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<std::complex<double>>(char transa, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda,
                                       const std::complex<double> *x, int incx, std::complex<double> beta, std::complex<double> *y,
                                       int incy) {
    zgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

/*!
 * Performs symmetric matrix diagonalization.
 */
auto ssyev(char job, char uplo, int n, float *a, int lda, float *w, float *work, int lwork) -> int;
auto dsyev(char job, char uplo, int n, double *a, int lda, double *w, double *work, int lwork) -> int;

template <typename T>
auto syev(char job, char uplo, int n, T *a, int lda, T *w, T *work, int lwork) -> int;

template <>
inline auto syev<float>(char job, char uplo, int n, float *a, int lda, float *w, float *work, int lwork) -> int {
    return ssyev(job, uplo, n, a, lda, w, work, lwork);
}
template <>
inline auto syev<double>(char job, char uplo, int n, double *a, int lda, double *w, double *work, int lwork) -> int {
    return dsyev(job, uplo, n, a, lda, w, work, lwork);
}

/*!
 * Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 */
auto cheev(char job, char uplo, int n, std::complex<float> *a, int lda, float *w, std::complex<float> *work, int lwork, float *rwork)
    -> int;
auto zheev(char job, char uplo, int n, std::complex<double> *a, int lda, double *w, std::complex<double> *work, int lwork, double *rworl)
    -> int;

template <typename T>
auto heev(char job, char uplo, int n, std::complex<T> *a, int lda, T *w, std::complex<T> *work, int lwork, T *rwork) -> int;

template <>
inline auto heev<float>(char job, char uplo, int n, std::complex<float> *a, int lda, float *w, std::complex<float> *work, int lwork,
                        float *rwork) -> int {
    return cheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

template <>
inline auto heev<double>(char job, char uplo, int n, std::complex<double> *a, int lda, double *w, std::complex<double> *work, int lwork,
                         double *rwork) -> int {
    return zheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

/*!
 * Computes the solution to system of linear equations A * x = B for general
 * matrices.
 */
auto sgesv(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb) -> int;
auto dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) -> int;
auto cgesv(int n, int nrhs, std::complex<float> *a, int lda, int *ipiv, std::complex<float> *b, int ldb) -> int;
auto zgesv(int n, int nrhs, std::complex<double> *a, int lda, int *ipiv, std::complex<double> *b, int ldb) -> int;

template <typename T>
auto gesv(int n, int nrhs, T *a, int lda, int *ipiv, T *b, int ldb) -> int;

template <>
inline auto gesv<float>(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb) -> int {
    return sgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<double>(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) -> int {
    return dgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<std::complex<float>>(int n, int nrhs, std::complex<float> *a, int lda, int *ipiv, std::complex<float> *b, int ldb) -> int {
    return cgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<std::complex<double>>(int n, int nrhs, std::complex<double> *a, int lda, int *ipiv, std::complex<double> *b, int ldb)
    -> int {
    return zgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

void dscal(int n, double alpha, double *vec, int inc);

auto ddot(int n, const double *x, int incx, const double *y, int incy) -> double;

void daxpy(int n, double alpha_x, const double *x, int inc_x, double *y, int inc_y);

/*!
 * Performs a rank-1 update of a general matrix.
 *
 * The ?ger routines perform a matrix-vector operator defined as
 *    A := alpha*x*y' + A,
 * where:
 *   alpha is a scalar
 *   x is an m-element vector,
 *   y is an n-element vector,
 *   A is an m-by-n general matrix
 */
void dger(int m, int n, double alpha, const double *x, int inc_x, const double *y, int inc_y, double *a, int lda);

/*!
 * Computes the LU factorization of a general M-by-N matrix A
 * using partial pivoting with row interchanges.
 *
 * The factorization has the form
 *   A = P * L * U
 * where P is a permutation matri, L is lower triangular with
 * unit diagonal elements (lower trapezoidal if m > n) and U is upper
 * triangular (upper trapezoidal if m < n).
 *
 */
auto dgetrf(int, int, double *, int, int *) -> int;

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf
 *
 * Returns INFO
 *   0 if successful
 *  <0 the (-INFO)-th argument has an illegal value
 *  >0 U(INFO, INFO) is exactly zero; the matrix is singular
 */
auto dgetri(int, double *, int, const int *, double *, int) -> int;

/*!
 * Return the value of the 1-norm, Frobenius norm, infinity-norm, or the
 * largest absolute value of any element of a general rectangular matrix
 */
auto slange(char norm_type, int m, int n, const float *A, int lda, float *work) -> float;
auto dlange(char norm_type, int m, int n, const double *A, int lda, double *work) -> double;
auto clange(char norm_type, int m, int n, const std::complex<float> *A, int lda, float *work) -> float;
auto zlange(char norm_type, int m, int n, const std::complex<double> *A, int lda, double *work) -> double;

template <typename T>
auto lange(char norm_type, int m, int n, const T *A, int lda, complex_type_t<T> *work) -> complex_type_t<T>;

template <>
inline auto lange<float>(char norm_type, int m, int n, const float *A, int lda, float *work) -> float {
    return slange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<double>(char norm_type, int m, int n, const double *A, int lda, double *work) -> double {
    return dlange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<std::complex<float>>(char norm_type, int m, int n, const std::complex<float> *A, int lda, float *work) -> float {
    return clange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<std::complex<double>>(char norm_type, int m, int n, const std::complex<double> *A, int lda, double *work) -> double {
    return zlange(norm_type, m, n, A, lda, work);
}

void slassq(int n, const float *x, int incx, float *scale, float *sumsq);
void dlassq(int n, const double *x, int incx, double *scale, double *sumsq);
void classq(int n, const std::complex<float> *x, int incx, float *scale, float *sumsq);
void zlassq(int n, const std::complex<double> *x, int incx, double *scale, double *sumsq);

template <typename T>
void lassq(int n, const T *x, int incx, complex_type_t<T> *scale, complex_type_t<T> *sumsq);

template <>
inline void lassq<float>(int n, const float *x, int incx, float *scale, float *sumsq) {
    slassq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<double>(int n, const double *x, int incx, double *scale, double *sumsq) {
    dlassq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<std::complex<float>>(int n, const std::complex<float> *x, int incx, float *scale, float *sumsq) {
    classq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<std::complex<double>>(int n, const std::complex<double> *x, int incx, double *scale, double *sumsq) {
    zlassq(n, x, incx, scale, sumsq);
}

/*!
 * Computes the singular value decomposition of a general rectangular
 * matrix using a divide and conquer method.
 */
auto dgesdd(char, int, int, double *, int, double *, double *, int, double *, int, double *, int, int *) -> int;

/*!
 * Computes the Schur Decomposition of a Matrix
 * (Used in Lyapunov Solves)
 */
auto dgees(char jobvs, int n, double* a, int lda, int* sdim, double* wr, double* wi, double* vs, int ldvs) -> int;

/*!
 * Sylvester Solve
 * (Used in Lyapunov Solves)
 */
auto dtrsyl(char trana, char tranb, int isgn, int m, int n, const double* a, int lda, const double* b, int ldb, double* c, int ldc, double* scale) -> int;

} // namespace einsums::blas