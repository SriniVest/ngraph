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

#include <memory>
#include <vector>
#include <iostream>

#include "ngraph/op/greater.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/interpreter/exec_node.hpp"
#include "ngraph/runtime/reference/greater.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class GreaterExec;
        }
    }
}

class ngraph::runtime::interpreter::GreaterExec : public ExecNode
{
public:
    static ExecNode create(const ngraph::Node* node)
    {
        std::cout << "create Greater" << std::endl;
        return GreaterExec(node);
    }

    GreaterExec(const ngraph::Node* node)
        : m_node{dynamic_cast<const ngraph::op::Greater*>(node)}
    {
        (void)m_node; // Silence compiler warning

        std::cout << "Greater ctor" << std::endl;
    }

    template <typename T>
    void execute(const std::vector<std::shared_ptr<HostTensorView>>& out,
                 const std::vector<std::shared_ptr<HostTensorView>>& args)
    {
        std::cout << "execute Greater" << std::endl;
    }

private:
    const ngraph::op::Greater* m_node;
};
