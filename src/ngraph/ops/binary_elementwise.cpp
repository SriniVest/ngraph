// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <memory>

#include "ngraph/log.hpp"
#include "ngraph/ops/op.hpp"

using namespace std;
using namespace ngraph;

op::BinaryElementwise::BinaryElementwise(
    const std::string& node_type,
    std::function<const element::Type&(const element::Type&, const element::Type&)>
        element_type_function,
    const std::shared_ptr<Node>& arg0,
    const std::shared_ptr<Node>& arg1)
    : RequiresTensorViewArgs(node_type, Nodes{arg0, arg1})
{
    auto arg0_tensor_type = get_inputs().at(0).get_tensor_view_type();
    auto arg1_tensor_type = get_inputs().at(1).get_tensor_view_type();
    if (arg0_tensor_type->get_shape() != arg1_tensor_type->get_shape())
    {
        throw ngraph_error("Arguments must have the same tensor view shape");
    }

    const element::Type& result_element_type = element_type_function(
        arg0_tensor_type->get_element_type(), arg1_tensor_type->get_element_type());

    set_value_type_checked(
        make_shared<TensorViewType>(result_element_type, arg0_tensor_type->get_shape()));
}