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

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class ExecNode;
        }
    }
}

class ngraph::runtime::interpreter::ExecNode
{
public:
    static ExecNode create_exec(const Node* node);

    virtual void execute_(const std::vector<std::shared_ptr<HostTensorView>>& out,
                          const std::vector<std::shared_ptr<HostTensorView>>& args) = 0;

private:
    using create_t = std::function<ExecNode(const Node*)>;
    static std::unordered_map<std::type_index, create_t> s_list;
};
