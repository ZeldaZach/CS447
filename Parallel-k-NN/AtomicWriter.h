//
// Created by Zachary Halpern on 2019-03-09.
//

#ifndef PARALLEL_K_NN_ATOMICWRITER_H
#define PARALLEL_K_NN_ATOMICWRITER_H

#include <fstream>
#include <iostream>
#include <sstream>

class AtomicWriter
{
    std::ostringstream st;
    std::ostream &stream;

public:
    AtomicWriter(std::ostream &s = std::cout) : stream(s)
    {
    }
    template <typename T> AtomicWriter &operator<<(T const &t)
    {
        st << t;
        return *this;
    }
    AtomicWriter &operator<<(std::ostream &(*f)(std::ostream &))
    {
        st << f;
        return *this;
    }
    ~AtomicWriter()
    {
        stream << st.str();
    }
};

#endif // PARALLEL_K_NN_ATOMICWRITER_H
