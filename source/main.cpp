#include <complex>
#include <iostream>

#include "tensor.h"

int main() {

    container::Tensor t1(container::DataType::DT_INT, {2, 3, 4});
    container::Tensor t2(
            container::DataTypeToEnum<std::complex<double>>::value,
            container::DeviceTypeToEnum<container::DEVICE_CPU>::value,
            container::TensorShape({3, 4}));

    auto * t1_data = t1.data<int>();
    auto * t2_data = t2.data<std::complex<double>>();

    for (int ii = 0; ii < t1.NumElements(); ii++) {
        t1_data[ii] = ii;
    }
    for (int ii = 0; ii < t2.NumElements(); ii++) {
        t2_data[ii] = {18.2222, -3232.10889};
    }

    container::Tensor t3(t1_data, t1.data_type(), t1.device_type(), t1.shape());
    container::Tensor t4 = t2.to_device<container::DEVICE_GPU>();
    container::Tensor t5 = t4.cast<std::complex<float>>().to_device<container::DEVICE_CPU>();
    container::Tensor t6 = t1.slice({0, 1, 1}, {2, 2, 2});
    t5.zero();
    t5.reshape({6, 2});

    std::vector<container::Tensor*> tensors {&t1, &t2, &t3, &t4, &t5, &t6};

    for (const container::Tensor* t : tensors) {
        std::cout << *t << std::endl;
    }
    // TODO:
    // Add some math operations,
    // Unit test
}