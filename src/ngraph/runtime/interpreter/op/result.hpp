/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the \"License\");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an \"AS IS\" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "ngraph/op/result.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/interpreter/exec_node.hpp"
#include "ngraph/runtime/reference/result.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            template <typename T>
            class ResultExec;
        }
    }
}

template <typename T>
class ngraph::runtime::interpreter::ResultExec : public ExecNode<T>
{
public:
    static ExecNode<T> create(const ngraph::Node* node)
    {
        std::cout << "create Result" << std::endl;
        return ResultExec(node);
    }

    ResultExec(const ngraph::Node* node)
        : m_node{dynamic_cast<const ngraph::op::Result*>(node)}
    {
        (void)m_node; // Silence compiler warning

        std::cout << "Result ctor" << std::endl;
    }

    void execute(const std::vector<std::shared_ptr<HostTensorView>>& out,
                 const std::vector<std::shared_ptr<HostTensorView>>& args) override
    {
        std::cout << "execute Result" << std::endl;
    }

private:
    const ngraph::op::Result* m_node;
};
