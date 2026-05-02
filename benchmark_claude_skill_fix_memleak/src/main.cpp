#include "classes.hpp"

#include <iostream>
#include <string.h>
#include <utility>

void leak_memory()
{
    Foo f;
}

void do_not_leak_memory()
{
    Foo f;
    Bar b(std::move(f));
}

int main(int argc, char **argv)
{
    if (argc == 1) {
        std::cout << "Usage: " << argv[0] << " <leak | no-leak>\n";
        return 0;
    }

    if (strcmp(argv[1], "leak") == 0) {
        leak_memory();
    } else if (strcmp(argv[1], "no-leak") == 0) {
        do_not_leak_memory();
    } else {
        std::cerr << "Usage: " << argv[0] << " <leak | no-leak>\n";
        return 1;
    }

    return 0;
}
