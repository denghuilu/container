#include <complex>
#include <iostream>

#include "tensor.h"

int main() {

    container::Tensor t1(container::DataType::DT_FLOAT, {2, 3, 4});
    container::Tensor t2(
            container::DataTypeToEnum<std::complex<double>>::value,
            container::DeviceTypeToEnum<container::DEVICE_CPU>::value,
            container::TensorShape({3, 4}));

    auto * t1_data = t1.data<float>();
    auto * t2_data = t2.data<std::complex<double>>();

    for (int ii = 0; ii < t1.NumElements(); ii++) {
        t1_data[ii] = 1.0;
    }
    for (int ii = 0; ii < t2.NumElements(); ii++) {
        t2_data[ii] = {18.2222, -3232.10889};
    }

    container::Tensor t3(t1_data, t1.data_type(), t1.device_type(), t1.shape());
    container::Tensor t4 = t2.to_device<container::DEVICE_GPU>();
    container::Tensor t5 = t4.cast<std::complex<float>>().to_device<container::DEVICE_CPU>();
    t5.reshape({-1, 4});

    std::cout << t1 << std::endl;
    std::cout << "NumElements:\t" << t1.NumElements() << std::endl;
    std::cout << "TensorShape:\t" << t1.shape() << std::endl;
    std::cout << "DataType:\t" << t1.data_type() << std::endl;
    std::cout << "MemoryType:\t" << t1.device_type() << std::endl;
    std::cout << "Owns memory? :\t" << t1.buffer().OwnsMemory() << std::endl;
    std::cout << std::endl;

    std::cout << t2 << std::endl;
    std::cout << "NumElements:\t" << t2.NumElements() << std::endl;
    std::cout << "TensorShape:\t" << t2.shape() << std::endl;
    std::cout << "DataType:\t" << t2.data_type() << std::endl;
    std::cout << "MemoryType:\t" << t2.device_type() << std::endl;
    std::cout << "Owns memory? :\t" << t2.buffer().OwnsMemory() << std::endl;
    std::cout << std::endl;

    t3.zero();
    std::cout << t3 << std::endl;
    std::cout << "NumElements:\t" << t3.NumElements() << std::endl;
    std::cout << "TensorShape:\t" << t3.shape() << std::endl;
    std::cout << "DataType:\t" << t3.data_type() << std::endl;
    std::cout << "MemoryType:\t" << t3.device_type() << std::endl;
    std::cout << "Owns memory? :\t" << t3.buffer().OwnsMemory() << std::endl;
    std::cout << std::endl;

    std::cout << t4 << std::endl;
    std::cout << "NumElements:\t" << t4.NumElements() << std::endl;
    std::cout << "TensorShape:\t" << t4.shape() << std::endl;
    std::cout << "DataType:\t" << t4.data_type() << std::endl;
    std::cout << "MemoryType:\t" << t4.device_type() << std::endl;
    std::cout << "Owns memory? :\t" << t4.buffer().OwnsMemory() << std::endl;
    std::cout << std::endl;

    t5.zero();
    std::cout << t5 << std::endl;
    std::cout << "NumElements:\t" << t5.NumElements() << std::endl;
    std::cout << "TensorShape:\t" << t5.shape() << std::endl;
    std::cout << "DataType:\t" << t5.data_type() << std::endl;
    std::cout << "MemoryType:\t" << t5.device_type() << std::endl;
    std::cout << "Owns memory? :\t" << t5.buffer().OwnsMemory() << std::endl;

    // TODO:
    // Add some math operations,
    // Unit test
}