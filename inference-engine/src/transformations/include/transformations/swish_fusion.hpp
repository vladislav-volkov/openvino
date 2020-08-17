// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "itt.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SwishFusion;
class TRANSFORMATIONS_API SwishFusionWithSigmoid;
class TRANSFORMATIONS_API SwishFusionWithSigmoidWithBeta;
class TRANSFORMATIONS_API SwishFusionWithBeta;
class TRANSFORMATIONS_API SwishFusionWithoutBeta;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief SwishFusion transformation replaces various sub-graphs with a Swish op.
 */
class ngraph::pass::SwishFusion: public ngraph::pass::GraphRewrite {
public:
    SwishFusion() {
        add_matcher<ngraph::pass::SwishFusionWithSigmoid>();
        add_matcher<ngraph::pass::SwishFusionWithSigmoidWithBeta>();
        add_matcher<ngraph::pass::SwishFusionWithBeta>();
        add_matcher<ngraph::pass::SwishFusionWithoutBeta>();
    }

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::nGraphPass_LT, "ngraph::pass::SwishFusion");
        return GraphRewrite::run_on_function(f);
    }
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SwishFusionWithSigmoid replaces a sub-graphs x * Sigmoid(x) with a Swish op.
 */
 class ngraph::pass::SwishFusionWithSigmoid: public ngraph::pass::MatcherPass {
public:
    SwishFusionWithSigmoid();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SwishFusionWithSigmoid replaces a sub-graphs x * Sigmoid(x * beta) with a Swish op.
 */
class ngraph::pass::SwishFusionWithSigmoidWithBeta: public ngraph::pass::MatcherPass {
public:
    SwishFusionWithSigmoidWithBeta();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SwishFusionWithSigmoid replaces a sub-graphs x / (1.0 + exp(-x * beta)) with a Swish op.
 */
class ngraph::pass::SwishFusionWithBeta: public ngraph::pass::MatcherPass {
public:
    SwishFusionWithBeta();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SwishFusionWithSigmoid replaces a sub-graphs x / (1.0 + exp(-x)) with a Swish op.
 */
class ngraph::pass::SwishFusionWithoutBeta: public ngraph::pass::MatcherPass {
public:
    SwishFusionWithoutBeta();
};
