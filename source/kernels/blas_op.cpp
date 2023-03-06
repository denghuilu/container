#include "blas_op.h"

namespace container {
namespace op {

//// CPU specialization of actual computation.
//template <typename FPTYPE>
//struct zdot_real_op<FPTYPE, DEVICE_CPU> {
//    FPTYPE operator() (
//            const DEVICE_CPU* d,
//            const int& dim,
//            const std::complex<FPTYPE>* psi_L,
//            const std::complex<FPTYPE>* psi_R,
//            const bool reduce)
//    {
//        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//        // qianrui modify 2021-3-14
//        // Note that  ddot_(2*dim,a,1,b,1) = REAL( zdotc_(dim,a,1,b,1) )
//        const FPTYPE* pL = reinterpret_cast<const FPTYPE*>(psi_L);
//        const FPTYPE* pR = reinterpret_cast<const FPTYPE*>(psi_R);
//        FPTYPE result = BlasConnector::dot(2 * dim, pL, 1, pR, 1);
//        if (reduce) {
//            Parallel_Reduce::reduce_double_pool(result);
//        }
//        return result;
//    }
//};

template <typename FPTYPE>
struct scal_op<FPTYPE, DEVICE_CPU> {
    void operator()(
            const int &N,
            const std::complex<FPTYPE> *alpha,
            std::complex<FPTYPE> *X,
            const int &incx)
    {
        BlasConnector::scal(N, *alpha, X, incx);
    }
};

template <typename FPTYPE>
struct gemv_op<FPTYPE, DEVICE_CPU> {
    void operator()(
            const DEVICE_CPU *d,
            const char &trans,
            const int &m,
            const int &n,
            const std::complex<FPTYPE> *alpha,
            const std::complex<FPTYPE> *A,
            const int &lda,
            const std::complex<FPTYPE> *X,
            const int &incx,
            const std::complex<FPTYPE> *beta,
            std::complex<FPTYPE> *Y,
            const int &incy)
    {
        BlasConnector::gemv(trans, m, n, *alpha, A, lda, X, incx, *beta, Y, incy);
    }
};

template <typename FPTYPE>
struct axpy_op<FPTYPE, DEVICE_CPU> {
    void operator()(
            const DEVICE_CPU * /*ctx*/,
            const int &dim,
            const std::complex<FPTYPE> *alpha,
            const std::complex<FPTYPE> *X,
            const int &incX,
            std::complex<FPTYPE> *Y,
            const int &incY)
    {
        BlasConnector::axpy(dim, *alpha, X, incX, Y, incY);
    }
};

template <typename FPTYPE>
struct gemm_op<FPTYPE, DEVICE_CPU> {
    void operator()(
            const DEVICE_CPU * /*ctx*/,
            const char &transa,
            const char &transb,
            const int &m,
            const int &n,
            const int &k,
            const std::complex<FPTYPE> *alpha,
            const std::complex<FPTYPE> *a,
            const int &lda,
            const std::complex<FPTYPE> *b,
            const int &ldb,
            const std::complex<FPTYPE> *beta,
            std::complex<FPTYPE> *c,
            const int &ldc)
    {
        BlasConnector::gemm(transb, transa, n, m, k, *alpha, b, ldb, a, lda, *beta, c, ldc);
    }
};

// Explicitly instantiate functors for the types of functor registered.
template struct scal_op<float, DEVICE_CPU>;
template struct axpy_op<float, DEVICE_CPU>;
template struct gemv_op<float, DEVICE_CPU>;
template struct gemm_op<float, DEVICE_CPU>;

template struct scal_op<double, DEVICE_CPU>;
template struct axpy_op<double, DEVICE_CPU>;
template struct gemv_op<double, DEVICE_CPU>;
template struct gemm_op<double, DEVICE_CPU>;

} // namespace op
} // namespace container