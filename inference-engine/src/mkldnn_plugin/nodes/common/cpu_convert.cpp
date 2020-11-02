// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_convert.h"
#include "cpu_memcpy.h"
#include <mkldnn_subset.h>
#include <type_traits>
#include <tuple>
#include <ie_parallel.hpp>

using namespace InferenceEngine;

namespace {

template<typename srcType, typename dstType>
void convert(void *srcPtr, void *dstPtr, const size_t size) {
    if (std::is_same<srcType, dstType>::value) {
        cpu_memcpy(dstPtr, srcPtr, size*sizeof(dstType));
    } else {
        const srcType *srcData = reinterpret_cast<const srcType *>(srcPtr);
        dstType *dstData = reinterpret_cast<dstType *>(dstPtr);

        parallel_for(size, [&](size_t i) {
            dstData[i] = static_cast<dstType>(srcData[i]);
        });
    }
}

struct ConvertContext {
    void *srcPtr;
    void *dstPtr;
    size_t size;
    bool converted;
};

template<typename T>
struct Convert {
    using src_t = typename std::tuple_element<0, T>::type;
    using dst_t = typename std::tuple_element<1, T>::type;

    void operator()(ConvertContext & ctx) {
        convert<src_t, dst_t>(ctx.srcPtr, ctx.dstPtr, ctx.size);
        ctx.converted = true;
    }
};

}   // namespace

#define MKLDNN_CNV(ST, DT) MKLDNN_CASE2(Precision::ST, Precision::DT, PrecisionTrait<Precision::ST>::value_type, PrecisionTrait<Precision::DT>::value_type)

void cpu_convert(void *srcPtr, void *dstPtr, Precision srcPrc, Precision dstPrc, const size_t size) {
    if (srcPrc == dstPrc) {
        cpu_memcpy(dstPtr, srcPtr, size*dstPrc.size());
        return;
    }

    ConvertContext ctx = { srcPtr, dstPtr, size, false };

    MKLDNN_SWITCH(Convert, ctx, std::tie(srcPrc, dstPrc),
    MKLDNN_CNV(U8, I8),    MKLDNN_CNV(U8, U16),   MKLDNN_CNV(U8, I16),   MKLDNN_CNV(U8, I32),
    MKLDNN_CNV(U8, U64),   MKLDNN_CNV(U8, I64),   MKLDNN_CNV(U8, FP32),  MKLDNN_CNV(U8, BOOL),
    MKLDNN_CNV(I8, U8),    MKLDNN_CNV(I8, U16),   MKLDNN_CNV(I8, I16),   MKLDNN_CNV(I8, I32),
    MKLDNN_CNV(I8, U64),   MKLDNN_CNV(I8, I64),   MKLDNN_CNV(I8, FP32),  MKLDNN_CNV(I8, BOOL),
    MKLDNN_CNV(U16, U8),   MKLDNN_CNV(U16, I8),   MKLDNN_CNV(U16, I16),  MKLDNN_CNV(U16, I32),
    MKLDNN_CNV(U16, U64),  MKLDNN_CNV(U16, I64),  MKLDNN_CNV(U16, FP32), MKLDNN_CNV(U16, BOOL),
    MKLDNN_CNV(I16, U8),   MKLDNN_CNV(I16, I8),   MKLDNN_CNV(I16, U16),  MKLDNN_CNV(I16, I32),
    MKLDNN_CNV(I16, U64),  MKLDNN_CNV(I16, I64),  MKLDNN_CNV(I16, FP32), MKLDNN_CNV(I16, BOOL),
    MKLDNN_CNV(I32, U8),   MKLDNN_CNV(I32, I8),   MKLDNN_CNV(I32, U16),  MKLDNN_CNV(I32, I16),
    MKLDNN_CNV(I32, U64),  MKLDNN_CNV(I32, I64),  MKLDNN_CNV(I32, FP32), MKLDNN_CNV(I32, BOOL),
    MKLDNN_CNV(U64, U8),   MKLDNN_CNV(U64, I8),   MKLDNN_CNV(U64, U16),  MKLDNN_CNV(U64, I16),
    MKLDNN_CNV(U64, I32),  MKLDNN_CNV(U64, I64),  MKLDNN_CNV(U64, FP32), MKLDNN_CNV(U64, BOOL),
    MKLDNN_CNV(I64, U8),   MKLDNN_CNV(I64, I8),   MKLDNN_CNV(I64, U16),  MKLDNN_CNV(I64, I16),
    MKLDNN_CNV(I64, I32),  MKLDNN_CNV(I64, U64),  MKLDNN_CNV(I64, FP32), MKLDNN_CNV(I64, BOOL),
    MKLDNN_CNV(FP32, U8),  MKLDNN_CNV(FP32, I8),  MKLDNN_CNV(FP32, U16), MKLDNN_CNV(FP32, I16),
    MKLDNN_CNV(FP32, I32), MKLDNN_CNV(FP32, U64), MKLDNN_CNV(FP32, I64), MKLDNN_CNV(FP32, BOOL),
    MKLDNN_CNV(BOOL, U8),  MKLDNN_CNV(BOOL, I8),  MKLDNN_CNV(BOOL, U16), MKLDNN_CNV(BOOL, I16),
    MKLDNN_CNV(BOOL, I32), MKLDNN_CNV(BOOL, U64), MKLDNN_CNV(BOOL, I64), MKLDNN_CNV(BOOL, FP32));

    if (!ctx.converted)
        THROW_IE_EXCEPTION << "cpu_convert can't convert from: " << srcPrc << " precision to: " << dstPrc;
}
