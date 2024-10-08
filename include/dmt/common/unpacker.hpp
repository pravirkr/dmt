#pragma once

#include <memory>
#include <string>

#include "dmt/common/types.hpp"

class DataUnpackerImplBase {
public:
    DataUnpackerImplBase()                                       = default;
    DataUnpackerImplBase(const DataUnpackerImplBase&)            = delete;
    DataUnpackerImplBase& operator=(const DataUnpackerImplBase&) = delete;
    DataUnpackerImplBase(DataUnpackerImplBase&&)                 = delete;
    DataUnpackerImplBase& operator=(DataUnpackerImplBase&&)      = delete;
    virtual ~DataUnpackerImplBase()                              = default;

    virtual void unpack_and_padd_uint8(const uint8_t* __restrict__ data_in,
                                       SizeType in_size,
                                       ComplexType* __restrict__ data_p1,
                                       ComplexType* __restrict__ data_p2,
                                       SizeType out_size) const = 0;
    virtual void unpack_and_padd_int8(const int8_t* __restrict__ data_in,
                                      SizeType in_size,
                                      ComplexType* __restrict__ data_p1,
                                      ComplexType* __restrict__ data_p2,
                                      SizeType out_size) const  = 0;
};

class DataUnpacker {
public:
    DataUnpacker(SizeType nsub,
                 SizeType nbin,
                 SizeType noverlap,
                 SizeType nfft,
                 const std::string& in_order);

    DataUnpacker(const DataUnpacker&)            = delete;
    DataUnpacker& operator=(const DataUnpacker&) = delete;
    DataUnpacker(DataUnpacker&&)                 = default;
    DataUnpacker& operator=(DataUnpacker&&)      = default;
    ~DataUnpacker()                              = default;

    template <typename DataType>
    void unpack_and_padd(const DataType* __restrict__ data_in,
                         SizeType in_size,
                         ComplexType* __restrict__ data_p1,
                         ComplexType* __restrict__ data_p2,
                         SizeType out_size) const;

private:
    std::unique_ptr<DataUnpackerImplBase> m_pimpl;
};

// Explicit template instantiations declaration
extern template void
DataUnpacker::unpack_and_padd<uint8_t>(const uint8_t* __restrict__,
                                       SizeType,
                                       ComplexType* __restrict__,
                                       ComplexType* __restrict__,
                                       SizeType) const;

extern template void
DataUnpacker::unpack_and_padd<int8_t>(const int8_t* __restrict__,
                                      SizeType,
                                      ComplexType* __restrict__,
                                      ComplexType* __restrict__,
                                      SizeType) const;