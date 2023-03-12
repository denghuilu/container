#include <complex>
#include <iostream>

#include "tensor.h"
#include "tensor_map.h"
#include "kernels/einsum_op.h"

int main() {

    container::Tensor t1(container::DataType::DT_DOUBLE, {2, 3, 4});
    container::Tensor t7(container::DataType::DT_INT, {2, 3, 4});
    container::Tensor t2(
            container::DataTypeToEnum<std::complex<double>>::value,
            container::DeviceTypeToEnum<container::DEVICE_CPU>::value,
            container::TensorShape({3, 4}));

    auto * t1_data = t1.data<double>();
    auto * t7_data = t7.data<int>();
    auto * t2_data = t2.data<std::complex<double>>();

    for (int ii = 0; ii < t1.NumElements(); ii++) {
        t1_data[ii] = ii * 0.0000001;
        t7_data[ii] = ii;
    }
    // check the output format.
    t1_data[12] = 100.10;
    t1_data[13] = 3300.100;
    t7_data[12] = 100;

    for (int ii = 0; ii < t2.NumElements(); ii++) {
        t2_data[ii] = {18.2222, -3232.10889};
    }
    t2_data[7] = {183.22221, -3232.10889};

    container::TensorMap t3(t1_data, t1.data_type(), t1.device_type(), t1.shape());
    container::Tensor t4 = t2.to_device<container::DEVICE_GPU>();
    container::Tensor t5 = t4.cast<std::complex<float>>().to_device<container::DEVICE_CPU>();
    container::Tensor t6 = t1.slice({0, 1, 1}, {2, 2, 2});
    t5.zero();
    t5.reshape({6, 2});

    t7.resize({3, 3, 4});

    std::vector<container::Tensor*> tensors {&t1, &t2, &t3, &t4, &t5, &t6, &t7};
    // std::vector<container::Tensor*> tensors {&t2};

    for (const container::Tensor* t : tensors) {
        std::cout << *t << std::endl;
    }

    // Do something really interested.
    auto t8 = container::op::einsum("mk,kn->mn", &t1, &t1);
    // TODO:
    // Add some math operations,
    // Unit test
}