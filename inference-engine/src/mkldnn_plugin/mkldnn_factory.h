// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "mkldnn_subset.h"
#include <string>
#include <functional>
#include <type_traits>
#include <unordered_map>

namespace MKLDNNPlugin {

template<typename Key, typename T>
class Factory;

template<typename Key, typename T, typename ...Args>
class Factory<Key, T(Args...)> {
    Factory(Factory const&) = delete;
    Factory& operator=(Factory const&) = delete;

public:
    using builder_t = std::function<T(Args...)>;

    Factory(const std::string & name)
        : name(name) {}

#ifdef MKLDNN_SUBSET
    #define registerNode(Name, key, Impl)       \
        MKLDNN_EXPAND(MKLDNN_CAT(registerImpl, MKLDNN_SCOPE_IS_ENABLED(MKLDNN_CAT(CC2MKLDNNPlugin_, Name)))<Impl>(key, MKLDNN_TOSTRING(Name)))

    template<typename Impl>
    void registerImpl0(const Key &, const char *) {
    }
#else
    #define registerNode(Name, key, Impl) registerImpl1<Impl>(key, MKLDNN_TOSTRING(Name))
#endif

    template<typename Impl>
    void registerImpl1(const Key & key, const char *typeName) {
#ifdef MKLDNN_SUBSET_FIND
        const std::string task_name = "REG$" + name + "$" + to_string(key) + "$" + typeName;
        OV_ITT_SCOPED_TASK(MKLDNNPlugin::internal::itt::domains::CC2MKLDNNPlugin,
                           openvino::itt::handle(task_name));
#endif
        builders[key] = [](Args... args) -> T {
            Impl *impl = new Impl(args...);
            return static_cast<T>(impl);
        };
    }

    T create(const Key & key, Args... args) {
        auto builder = builders.find(key);
        if (builder != builders.end()) {
#ifdef MKLDNN_SUBSET_FIND
            const std::string task_name = "CREATE$" + name + "$" + to_string(key);
            OV_ITT_SCOPED_TASK(MKLDNNPlugin::internal::itt::domains::CC2MKLDNNPlugin,
                            openvino::itt::handle(task_name));
#endif
            return builder->second(args...);
        }
        return nullptr;
    }

    template<typename Fn>
    void foreach(Fn fn) const {
        for (auto itm : builders)
            fn(itm);
    }

    size_t size() const noexcept {
        return builders.size();
    }

private:
    const std::string & to_string(const std::string & str) const noexcept {
        return str;
    }

    template<typename V,
             typename std::enable_if<std::is_enum<V>::value, bool>::type = true>
    std::string to_string(V val) const {
        return std::to_string(static_cast<int>(val));
    }

    template<typename V,
             typename std::enable_if<!std::is_enum<V>::value, bool>::type = true>
    std::string to_string(V val) const {
        return std::to_string(val);
    }

    using map_t = std::unordered_map<Key, builder_t>;

    const std::string name;
    map_t builders;
};

}   // namespace MKLDNNPlugin
