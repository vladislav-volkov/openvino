// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertShapeOf3;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertShapeOf3: public ngraph::pass::GraphRewrite {
public:
    ConvertShapeOf3() : GraphRewrite() {
        convert_shapeof3();
    }

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

private:
    void convert_shapeof3();
};
