#include <random>
#include <array>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cxxabi.h>
#include <typeinfo>



// Epsilon to use when comparing two FP numbers.
template <typename T> constexpr T eps;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-const-variable"
template <> constexpr float eps<float> = 1E-6;
// Use more precision for doubles.
template <> constexpr float eps<double> = 1E-15;
#pragma GCC diagnostic pop

// Precision to use for the computations.  Double is more accurate, single
// (float) is faster, generally.
using precision_t = float;
// Number of inputs to the softmax.
constexpr size_t N = 4;

// For convenience.
using uniform_dist_t = std::uniform_real_distribution<precision_t>;

// Engine to use, throughout.
std::default_random_engine eng;
// Use this if you want different results every run.
// std::default_random_engine eng(std::random_device{}());

// Print out type names, for debugging.
std::ostream &
operator<<(std::ostream &os, const std::type_info &ti) {
    int ec;
    const char *demangled_name = abi::__cxa_demangle(ti.name(), 0, 0, &ec);
    assert(ec == 0);
    os << demangled_name;
    free((void *) demangled_name);
    return os;
}

// Function to test FP equality, using epsilon.
template <typename T1, typename T2>
inline bool
approx_equal(const T1 &v1, const T2 &v2) {
    using T = decltype(v1 - v2);
    static_assert(std::is_floating_point_v<T>,
     "Can't compare integral types for approximate equality.");
    return std::abs(v1 - v2) < eps<T>;
}

// This is a functor that returns a comma+space every time it is called, except
// the first time.  Use it for outputting lists.
class SepGen {
    public:
        SepGen() {}
        const char *operator()() {
            if (!first) {
                return ", ";
            } else {
                first = false;
                return "";
            }
        }
    private:
        bool first = true;
};

// Array output operator.
template <typename T, size_t N>
std::ostream &
operator<<(std::ostream &os, const std::array<T, N> &a) {
    SepGen sep;
    for (const auto &e : a) {
        os << sep() << e;
    }
    return os;
}

template <typename T, size_t N>
auto
softmax(const std::array<T, N> &in) {

    static_assert(std::is_floating_point_v<T>, "Can't compute softmax with integral types.");

    // Use identity softmax(x) == softmax(x - C) to guard against under/overflow.
    const auto C = *std::max_element(in.begin(), in.end());
    auto out{in};
    T sum = 0;
    for (size_t i = 0; i < N; i++) {
        out[i] = std::exp(in[i] - C);
        sum += out[i];
    }
    /*
    for (size_t i = 0; i < N; i++) {
        out[i] = out[i]/sum;
    }
    */
    std::transform(out.begin(), out.end(), out.begin(), [sum](auto e) { return e/sum; });

    // Verify that it is a probability: Sums to 1 and all >= 0.
    assert(approx_equal(std::accumulate(out.begin(), out.end(), T(0)), 1));
    #ifndef NDEBUG
    std::for_each(out.begin(), out.end(), [](auto e) { assert(e >= 0); });
    #endif

    return out;
}

template <typename T, size_t N>
auto
softmax_ds(const std::array<T, N> &out, const std::array<T, N> &us) {

    std::array<T, N> sm_ds{};
    // std::cout << "SM ds start: " << sm_ds << std::endl;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            if (i == j) {
                sm_ds[j] += (out[i]*(1 - out[j]))*us[i];
            } else {
                sm_ds[j] += (-out[j]*out[i])*us[i];
            }
            /*
            printf("SM ds [%zu,%zu]: ", i, j);
            std::cout << sm_ds << std::endl;
            */
        }
    }

    /*
     * BUG!
    for (size_t i = 0; i < N; i++) {
        sm_ds[i] *= us[i];
    }
    */

    return sm_ds;
}

// This is the loss
template <typename T1, typename T2, size_t N>
auto
cross_entropy(const std::array<T1, N> &p, const std::array<T2, N> &q) {

    using promoted_t = decltype(p[0]*q[0]);

    static_assert(std::is_floating_point_v<promoted_t>,
     "Can't compute cross-entropy with two integral types.");

    promoted_t ce = 0;
    for (size_t i = 0; i < N; i++) {
        ce += p[i]*std::log(q[i]);
    }
    return -ce;
}

template <typename T1, typename T2, size_t N>
auto
cross_entropy_ds(const std::array<T1, N> &p, const std::array<T2, N> &q) {
    std::array<T2, N> ce_ds;
    for (size_t i = 0; i < N; i++) {
        ce_ds[i] = -p[i]/q[i];
    }
    return ce_ds;
}

template <typename T>
auto
train(const T &sm_in, const T &ce_td) {

    /*
     * Forward pass.
     */

    auto sm_out = softmax(sm_in);
    std::cout << "Softmax out: " << sm_out << std::endl;

    auto ce = cross_entropy(ce_td, sm_out);
    std::cout << "CE: " << ce << std::endl;

    /*
     * Backprop.
     */
    
    auto ce_ds = cross_entropy_ds(ce_td, sm_out);
    std::cout << "CE downstream: " << ce_ds << std::endl;

    // ce_ds now becomes softmax upstream derivative.
    auto sm_ds = softmax_ds(sm_out, ce_ds);
    std::cout << "SM downstream: " << sm_ds << std::endl;

    return std::pair{ce, sm_ds};
}

int
main() {

    // Generate inputs to the softmax, randomly.
    std::array<precision_t, N> sm_in;
    std::generate(sm_in.begin(),  sm_in.end(),
        // Distributions are cheap to construct, and have no state.
        []() { return uniform_dist_t{-10, 10}(eng); }
    );
    std::cout << "Softmax in: " << sm_in << std::endl;

    // Cross-entropy requires a true probability distribution.  Generate that
    // now.
    decltype(sm_in) ce_td;
    std::generate(ce_td.begin(),  ce_td.end(),
        []() { return uniform_dist_t{0, 10}(eng); }
    );
    // Normalize, to make it a probability.
    std::transform(ce_td.begin(), ce_td.end(), ce_td.begin(),
        [sum = std::accumulate(ce_td.begin(), ce_td.end(), typename decltype(ce_td)::value_type(0))](auto v) {
            return v/sum;
        }
    );
    std::cout << "CE true dist: " << ce_td << std::endl;
    // Verify that it sums to 1, since it is a probability.
    assert(approx_equal(std::accumulate(ce_td.begin(), ce_td.end(), precision_t(0)), 1));

    // Now train.  Does forward and backprop.
    auto res = train(sm_in, ce_td);
    const auto &ce(res.first);
    const auto &grad(res.second);

    std::cout << "CE: " << ce << std::endl;
    std::cout << "Gradient: " << grad << std::endl;

    /*
     * grad now says, when sm_in[i] changes by dh, then the loss (ce) will
     * change by grad[i]*dh.  Let's test that.  Unfortunately, laborious
     * validation is a requirement for numerical code.
     */

    std::cout << "==== TESTING" << std::endl;

    constexpr precision_t dh = .0001;

    // This will modify an input by dh, then recompute the loss.  It then
    // compares the loss that is approximated by using the gradient vs the
    // actual change in loss.  It restores the input.
    auto test = [&](size_t i, auto dh) {
        auto save = sm_in[i];
        sm_in[i] = sm_in[i] + dh;
        auto delta_actual = train(sm_in, ce_td).first - ce;
        auto delta_approx = grad[i]*dh; // Approximated using the gradient.
        sm_in[i] = save; // Restore the input.
        std::cout << "--> dh=" << dh;
        std::cout << ", ";
        std::cout << "delta_actual=" << delta_actual;
        std::cout << ", ";
        std::cout << "delta_approximated=" << delta_approx;
        std::cout << ", ";
        std::cout << "residual=" << delta_approx - delta_actual;
        std::cout << std::endl;
    };

    for (size_t i = 0; i < N; i++) {
        test(i, dh);
        test(i, -dh);
    }

    #if 0
    {
        std::cout << "TEST CE" << std::endl;
        std::array<double, 2> p{{.2, .8}}, q{{.3, .7}};
        std::cout << "CE p: " << p << std::endl;
        std::cout << "CE q: " << q << std::endl;
        std::cout << "CE: " << cross_entropy(p, q) << std::endl;
        std::cout << "CE ds: " << cross_entropy_ds(p, q) << std::endl;
    }

    {
        std::cout << "TEST SM" << std::endl;
        std::array<double, 2> in{{1, 2}};
        {
            auto out = softmax(in);
            auto ds = softmax_ds(out, {{1, 1}});
            std::cout << "SM in: " << in << std::endl;
            std::cout << "SM out: " << out << std::endl;
            std::cout << "SM ds: " << ds << std::endl;
        }

        // Change +.01
        in[0] += .01;
        {
            auto out = softmax(in);
            auto ds = softmax_ds(out, {{1, 1}});
            std::cout << "CHANGE" << std::endl;
            std::cout << "SM in: " << in << std::endl;
            std::cout << "SM out: " << out << std::endl;
        }
    }
    #endif
}
