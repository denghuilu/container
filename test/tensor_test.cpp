#include <gtest/gtest.h>

#include "../tensor.h"

TEST(TensorTest, Constructor) {
    // Test default constructor
    ABACUS::Tensor t1(DataType::DT_FLOAT, ABACUS::TensorShape({2, 3}));
    EXPECT_EQ(t1.dtype(), DataType::DT_FLOAT);
    EXPECT_EQ(t1.shape(), ABACUS::TensorShape({2, 3}));
    EXPECT_EQ(t1.NumElements(), 6);
    EXPECT_NE(t1.data(), nullptr);

    // Test constructor with allocator
    ABACUS::Allocator* allocator = new ABACUS::DefaultAllocator();
    ABACUS::Tensor t2(DataType::DT_INT32, ABACUS::TensorShape({3, 4}), allocator);
    EXPECT_EQ(t2.dtype(), DataType::DT_INT32);
    EXPECT_EQ(t2.shape(), ABACUS::TensorShape({3, 4}));
    EXPECT_EQ(t2.NumElements(), 12);
    EXPECT_NE(t2.data(), nullptr);
    EXPECT_EQ(t2.buffer().allocator(), allocator);

    // Test copy constructor
    ABACUS::Tensor t3(t1);
    EXPECT_EQ(t3.dtype(), DataType::DT_FLOAT);
    EXPECT_EQ(t3.shape(), ABACUS::TensorShape({2, 3}));
    EXPECT_EQ(t3.NumElements(), 6);
    EXPECT_NE(t3.data(), nullptr);
    EXPECT_NE(t3.data(), t1.data());
}

TEST(TensorTest, SizeOfType) {
    EXPECT_EQ(ABACUS::Tensor::SizeOfType(DataType::DT_FLOAT), sizeof(float));
    EXPECT_EQ(ABACUS::Tensor::SizeOfType(DataType::DT_INT32), sizeof(int32_t));
    EXPECT_EQ(ABACUS::Tensor::SizeOfType(DataType::DT_DOUBLE), sizeof(double));
    EXPECT_EQ(ABACUS::Tensor::SizeOfType(DataType::DT_COMPLEX), sizeof(std::complex<float>));
    EXPECT_EQ(ABACUS::Tensor::SizeOfType(DataType::DT_COMPLEX_DOUBLE), sizeof(std::complex<double>));
}
