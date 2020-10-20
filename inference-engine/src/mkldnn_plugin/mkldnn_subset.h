// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/*
        This file contains a useful API for analyzing MKLDNN plugin sources
    and enabling or disabling some regions.
        Three working modes are currently supported.
    * MKLDNN_SUBSET_FIND    This macro enables analysis mode for annotated code regions.
    *                       When the process completes, a new C++ header file is created
    *                       that contains macros for enabling active regions. This file
    *                       should be included in all analysed C++ files.
    * MKLDNN_SUBSET         This mode disables inactive areas of the code using the result
    *                       of the analysis step.
    * No definitions        The default behavior is keept if no MKLDNN_SUBSET * macros are defined,
    *                       i.e all features of the MKLDNN plugin are enabled.
    *
    * An example of using annotation:
    *
    *  I. Any C++ code block:
    *       MKLDNN_SCOPE(ScopeName,
    *           // Any C++ code.
    *           cout << "Hello world!";
    *       );
    *
    *  II. Template class instantiation using switch-case:
    *
    *    struct Context { ... };
    *
    *    template<typename T>
    *    struct SomeTemplateClass {
    *        void operator()(Context &context) {
    *           // Any C++ code.
    *           cout << "Hello world!";
    *        }
    *    };
    *
    *    auto key = Precision::U8;
    *    Context context;
    *
    *    MKLDNN_SWITCH(SomeTemplateClass, context, key,
    *        MKLDNN_CASE(Precision::U8, uint8_t),
    *        MKLDNN_CASE(Precision::I8, int8_t),
    *        MKLDNN_CASE(Precision::FP32, float));
    *
*/

#ifdef MKLDNN_SUBSET_FIND
#include "mkldnn_itt.h"
#include <string>
#endif

#include <utility>
#include <tuple>

namespace MKLDNNPlugin {

// Macros for names concatenation
#define MKLDNN_CAT_(x, y) x ## y
#define MKLDNN_CAT(x, y) MKLDNN_CAT_(x, y)
#define MKLDNN_CAT3(x1, x2, x3) MKLDNN_CAT(MKLDNN_CAT(x1, x2), x3)

// Expand macro argument
#define MKLDNN_EXPAND(x) x

// Macros for string conversion
#define MKLDNN_TOSTRING(...) MKLDNN_TOSTRING_(__VA_ARGS__)
#define MKLDNN_TOSTRING_(...) #__VA_ARGS__

#ifndef MKLDNN_SUBSET_FIND

namespace internal {

template<typename C, typename T>
struct case_wrapper {
    using type = T;
    const C value {};

    case_wrapper(C && val)
        : value(std::forward<C>(val))
    {}
};

template<typename T, typename C>
case_wrapper<C, T> make_case_wrapper(C && val) {
    return case_wrapper<C, T>(std::forward<C>(val));
}

template<template<typename...> typename Fn, typename Ctx, typename T, typename Case>
bool match(Ctx && ctx, T && val, Case && cs) {
    const bool is_matched = val == cs.value;
    if (is_matched)
        Fn<typename Case::type>()(std::forward<Ctx>(ctx));
    return is_matched;
}

template<template<typename...> typename Fn, typename Ctx, typename T, typename Case, typename ...Cases>
bool match(Ctx && ctx, T && val, Case && cs, Cases&&... cases) {
    if (match<Fn>(std::forward<Ctx>(ctx), std::forward<T>(val), std::forward<Case>(cs)))
        return true;
    return match<Fn>(std::forward<Ctx>(ctx), std::forward<T>(val), std::forward<Cases>(cases)...);
}

}   // namespace internal

#endif

#ifdef MKLDNN_SUBSET_FIND           // MKLDNN analysis

namespace internal {
namespace itt {
namespace domains {
    OV_ITT_DOMAIN(CC0MKLDNNPlugin); // Domain for simple scope surrounded by ifdefs
    OV_ITT_DOMAIN(CC1MKLDNNPlugin); // Domain for switch/cases
    OV_ITT_DOMAIN(CC2MKLDNNPlugin); // Domain for MKLDNN plugin factories
}   // namespace domains
}   // namespace itt

template<typename C, typename T>
struct case_wrapper {
    using type = T;
    const C value {};
    const char *name = nullptr;

    case_wrapper(C && val, const char *name)
        : value(std::forward<C>(val))
        , name(name)
    {}
};

template<typename T, typename C>
case_wrapper<C, T> make_case_wrapper(C && val, const char *name) {
    return case_wrapper<C, T>(std::forward<C>(val), name);
}

template<template<typename...> typename Fn, typename Ctx, typename T, typename Case>
bool match(char const *region, Ctx && ctx, T && val, Case && cs) {
    const bool is_matched = val == cs.value;
    if (is_matched) {
        OV_ITT_SCOPED_TASK(MKLDNNPlugin::internal::itt::domains::CC1MKLDNNPlugin, std::string(region) + ":" + cs.name);
        Fn<typename Case::type>()(std::forward<Ctx>(ctx));
    }
    return is_matched;
}

template<template<typename...> typename Fn, typename Ctx, typename T, typename Case, typename ...Cases>
bool match(char const *region, Ctx && ctx, T && val, Case && cs, Cases&&... cases) {
    if (match<Fn>(region, std::forward<Ctx>(ctx), std::forward<T>(val), std::forward<Case>(cs)))
        return true;
    return match<Fn>(region, std::forward<Ctx>(ctx), std::forward<T>(val), std::forward<Cases>(cases)...);
}

}   // namespace internal

#define MKLDNN_SCOPE(region, ...)                                                       \
    OV_ITT_SCOPED_TASK(MKLDNNPlugin::internal::itt::domains::CC0MKLDNNPlugin, MKLDNN_TOSTRING(region)); \
    __VA_ARGS__

#define MKLDNN_SWITCH(fn, ctx, val, ...)                                                \
    internal::match<fn>(MKLDNN_TOSTRING(fn), ctx, val, __VA_ARGS__);

#define MKLDNN_LBR (
#define MKLDNN_RBR )

#define MKLDNN_CASE(Case, Type)                                                         \
    internal::make_case_wrapper<Type>(Case, MKLDNN_TOSTRING(MKLDNN_CASE MKLDNN_LBR Case, Type MKLDNN_RBR))

#define MKLDNN_CASE2(Case1, Case2, Type1, Type2)                                        \
    internal::make_case_wrapper<std::tuple<Type1, Type2>>(                              \
        std::make_tuple(Case1, Case2),                                                  \
        MKLDNN_TOSTRING(MKLDNN_CASE2 MKLDNN_LBR Case1, Case2, Type1, Type2 MKLDNN_RBR))

#elif defined(MKLDNN_SUBSET)        // MKLDNN subset is used

// Placeholder for first macro argument
#define MKLDNN_SCOPE_ARG_PLACEHOLDER_1 0,

// This macro returns second argument, first argument is ignored
#define MKLDNN_SCOPE_SECOND_ARG(ignored, val, ...) val

// Return macro argument value
#define MKLDNN_SCOPE_IS_ENABLED(x) MKLDNN_SCOPE_IS_ENABLED1(x)

// Generate junk macro or {0, } sequence if val is 1
#define MKLDNN_SCOPE_IS_ENABLED1(val) MKLDNN_SCOPE_IS_ENABLED2(MKLDNN_SCOPE_ARG_PLACEHOLDER_##val)

// Return second argument from possible sequences {1, 0}, {0, 1, 0}
#define MKLDNN_SCOPE_IS_ENABLED2(arg1_or_junk) MKLDNN_SCOPE_SECOND_ARG(arg1_or_junk 1, 0)

// Scope is disabled
#define MKLDNN_SCOPE_0(...)

// Scope is enabled
#define MKLDNN_SCOPE_1(...) __VA_ARGS__

#define MKLDNN_SCOPE(region, ...)           \
    MKLDNN_EXPAND(MKLDNN_CAT(MKLDNN_SCOPE_, MKLDNN_SCOPE_IS_ENABLED(MKLDNN_CAT(CC0MKLDNNPlugin_, region)))(__VA_ARGS__))

// Switch is disabled
#define MKLDNN_SWITCH_0(fn, ctx, val)

// Switch is enabled
#define MKLDNN_SWITCH_1(fn, ctx, val) internal::match<fn>(ctx, val, MKLDNN_CAT3(CC1MKLDNNPlugin_, fn, _cases));

#define MKLDNN_SWITCH(fn, ctx, val, ...)         \
    MKLDNN_EXPAND(MKLDNN_CAT(MKLDNN_SWITCH_, MKLDNN_SCOPE_IS_ENABLED(MKLDNN_CAT(CC1MKLDNNPlugin_, fn)))(fn, ctx, val))

#define MKLDNN_CASE(Case, Type) internal::make_case_wrapper<Type>(Case)

#define MKLDNN_CASE2(Case1, Case2, Type1, Type2) internal::make_case_wrapper<std::tuple<Type1, Type2>>(std::make_tuple(Case1, Case2))

#else

#define MKLDNN_SCOPE(region, ...) __VA_ARGS__

#define MKLDNN_SWITCH(fn, ctx, val, ...)    \
    internal::match<fn>(ctx, val, __VA_ARGS__);

#define MKLDNN_CASE(Case, Type) internal::make_case_wrapper<Type>(Case)

#define MKLDNN_CASE2(Case1, Case2, Type1, Type2) internal::make_case_wrapper<std::tuple<Type1, Type2>>(std::make_tuple(Case1, Case2))

#endif

}   // namespace MKLDNNPlugin
