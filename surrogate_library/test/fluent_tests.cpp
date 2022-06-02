
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include "fluent.h"
using namespace phasm::fluent;

struct MyStruct {
    int x = 22;
    double y[3] = {1.0, 2.0, 3.0};
};

int my_global;
TEST_CASE("Basic fluent construction") {

    OpticBuilder builder;

    builder
    .local<int>("a")
        .primitive("a")
        .end()
    .global("my_global", &my_global)
        .primitive("b")
        .end()
    .local<MyStruct>("s")
        .accessor<int>([](MyStruct* s){return &(s->x);})
            .primitive("x")
            .end()
        .accessor<double>([](MyStruct* s){return s->y;})
            .primitives("y", {3})
            .end();

    REQUIRE(builder.get_callsite_vars().size() == 3);
    REQUIRE(builder.get_model_vars().size() == 4);

    builder.printOpticsTree();
    std::cout << "-----------" << std::endl;
    builder.printModelVars();

}




