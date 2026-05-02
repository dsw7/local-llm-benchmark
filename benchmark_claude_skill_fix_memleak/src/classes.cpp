#include "classes.hpp"

Foo::Foo()
{
    this->val = new int(5);
}

Bar::Bar(Foo &&other) noexcept
{
    this->val_ = other.val;
    other.val = nullptr;
}

Bar::~Bar()
{
    if (this->val_) {
        delete this->val_;
    }
}
