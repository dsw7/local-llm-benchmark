#pragma once

struct Foo {
    Foo();
    int *val = nullptr;
};

class Bar {
public:
    Bar(Foo &&other) noexcept;
    ~Bar();

private:
    int *val_ = nullptr;
};
