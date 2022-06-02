
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#ifndef SURROGATE_TOOLKIT_BINDINGS_H
#define SURROGATE_TOOLKIT_BINDINGS_H

#include "model_variable.h"
#include "optics.h"
#include "any_ptr.hpp"

namespace phasm {

struct CallSiteVariable {

    std::string name;
    phasm::any_ptr binding;
    std::vector<OpticBase*> optics_tree;
    std::vector<std::shared_ptr<ModelVariable>> model_vars;


    inline void captureAllTrainingInputs() {
        for (const auto& model_var : model_vars) {
            if (model_var->is_input) {
                model_var->captureTrainingInput(binding);
            }
        }
    }

    inline void captureAllTrainingOutputs() {
        for (const auto& model_var : model_vars) {
            if (model_var->is_output) {
                model_var->captureTrainingOutput(binding);
            }
        }
    }

    inline void captureAllInferenceInputs() {
        for (const auto& model_var : model_vars) {
            if (model_var->is_input) {
                model_var->captureInferenceInput(binding);
            }
        }
    }

    inline void publishAllInferenceOutputs() {
        for (const auto& model_var : model_vars) {
            if (model_var->is_output) {
                model_var->publishInferenceOutput(binding);
            }
        }
    }
};

} // namespace phasm


#endif //SURROGATE_TOOLKIT_BINDINGS_H
