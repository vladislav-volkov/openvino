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

class TRANSFORMATIONS_API ConvertMulAddToScaleShiftOrPower;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMulAddToScaleShiftOrPower: public ngraph::pass::GraphRewrite {
public:
    ConvertMulAddToScaleShiftOrPower() : GraphRewrite() {
        convert_mul_add_to_scaleshift_or_power();
    }

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

private:
    void convert_mul_add_to_scaleshift_or_power();
};

enum class CONVERSION_RESULT {
    SCALE_SHIFT,
    POWER,
    NONE
};

/*
 * check_constant function checks how given constant performs elementwise operation with given input
 * CONVERSION_RESULT has several types:
 *      SCALE_SHIFT - constant applies only per-channel
 *      POWER - constant applies as single value
 *      NONE - default return value
 */

TRANSFORMATIONS_API CONVERSION_RESULT
check_constant(const std::shared_ptr<ngraph::op::Constant> & constant, const ngraph::PartialShape & shape);
