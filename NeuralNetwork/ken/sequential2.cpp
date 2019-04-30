/*
 * In this code, downstream and upstream are relative to data flow direction.  So during the forward pass,
 * downstream is in the forward direction, while during back propagation, downstream is in the backward
 * direction.
 *
 * Also, currently train() and backprop() are currently implemented as separate functions.  I think it is
 * possible, and maybe cleaner, to instead implement them as one function.  On the return, the backprop is
 * done.
 */

#include <cmath>
#include <cassert>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>

using std::size_t;
using Precision = double;
constexpr double DELTA_H = 1E-5;
constexpr int PADDING = 2;
constexpr int FILTER_H = 5;
constexpr int FILTER_W = 5;

inline double
derivative_error(double n, double d) {
    return std::abs(n - d)/std::max(std::abs(n), std::abs(d));
}

// This holds a sequence of dimensions together in a single type.
template <size_t DP, size_t HP, size_t WP>
struct Dims {
    constexpr static size_t D = DP;
    constexpr static size_t H = HP;
    constexpr static size_t W = WP;
    constexpr static size_t N = D*H*W;
};

template <typename T, size_t D, size_t H, size_t W>
std::ostream &operator<<(std::ostream &os, const T (&a)[D][H][W]) {
    for (size_t h = 0; h < D; h++) {
        if (h > 0) {
            os << "----------" << std::endl;
        }
        for (size_t i = 0; i < H; i++) {
            for (size_t j = 0; j < W; j++) {
                if (j > 0) {
                    os << " ";
                }
                os << std::fixed << std::setprecision(7) << a[h][i][j];
            }
            os << "\n";
        }
    }
    return os;
}

/*
 * Array class:  This is a wrapper around native arrays to get range-checking.
 * It is similar to std::array, but more convenient for multi-dimensional arrays.
 */

// Forward declaration for output operators.
template <typename T, size_t D, size_t... Ds> class Array;

// Output operators for up to 4-D.
template <typename T, size_t D0>
std::ostream &
operator<<(std::ostream &os, const Array<T, D0> &a) {
    for (size_t i = 0; i < D0; i++) {
        if (i > 0) {
            os << " ";
        }
        os << std::fixed << std::setprecision(7) << a[i];
    }
    os << std::endl;
    return os;
}

template <typename T, size_t D1, size_t D0>
std::ostream &
operator<<(std::ostream &os, const Array<T, D1, D0> &a) {
    for (size_t i = 0; i < D1; i++) {
        os << std::fixed << std::setprecision(7) << a[i];
    }
    return os;
}

template <typename T, size_t D2, size_t D1, size_t D0>
std::ostream &
operator<<(std::ostream &os, const Array<T, D2, D1, D0> &a) {
    for (size_t h = 0; h < D2; h++) {
        os << "Matrix " << h << ":" << std::endl;
        os << a[h];
    }
    return os;
}

template <typename T, size_t D3, size_t D2, size_t D1, size_t D0>
std::ostream &
operator<<(std::ostream &os, const Array<T, D3, D2, D1, D0> &a) {
    for (size_t g = 0; g < D3; g++) {
        os << "Tensor " << g << ":" << std::endl;
        os << a[g];
    }
    return os;
}

// General definition of template.
template <typename T, size_t D, size_t... Ds>
class Array {
        friend std::ostream &operator<<<>(std::ostream &, const Array &);
    public:
        Array() = default;
        template <typename U>
        Array(const U &v) {
            *this = v;
        }
        Array<T, Ds...> &operator[](const size_t i) {
            assert(i < D);
            return array[i];
        }
        const Array<T, Ds...> &operator[](const size_t i) const {
            assert(i < D);
            return array[i];
        }
        template <typename... Ts>
        T &operator()(const size_t i, const Ts... rest) {
            return (*this)[i](rest...);
        }
        template <typename... Ts>
        const T &operator()(const size_t i, const Ts... rest) const {
            return (*this)[i](rest...);
        }
        template <typename U>
        Array &operator=(const U &v) {
            std::fill(std::begin(array), std::end(array), v);
            return *this;
        }
        template <typename U>
        Array &operator=(const U (&a)[D]) {
            std::copy(std::begin(a), std::end(a), std::begin(array));
            return *this;
        }
        Array<T, Ds...> *begin() { return &array[0]; }
        Array<T, Ds...> *end() { return &array[D]; }
        const Array<T, Ds...> *begin() const { return &array[0]; }
        const Array<T, Ds...> *end() const { return &array[D]; }
    private:
        Array<T, Ds...> array[D];
};

// Base case.
template <typename T, size_t D>
class Array<T, D> {
        friend std::ostream &operator<<<>(std::ostream &, const Array &);
    public:
        Array() = default;
        template <typename U>
        Array(const U &v) {
            *this = v;
        }
        T &operator[](const size_t i) {
            #ifndef NDEBUG
            if (i >= D) {
                std::cerr << "Index " << i << " beyond end of array of size " << D << "." << std::endl;
                assert(false);
                abort();
            }
            #endif
            return array[i];
        }
        const T&operator[](const size_t i) const {
            #ifndef NDEBUG
            if (i >= D) {
                std::cerr << "Index " << i << " beyond end of array of size " << D << "." << std::endl;
                assert(false);
                abort();
            }
            #endif
            return array[i];
        }
        T &operator()(const size_t i) {
            return (*this)[i];
        }
        const T &operator()(const size_t i) const {
            return (*this)[i];
        }
        template <typename U>
        Array &operator=(const Array<U, D> &a) {
            std::copy(std::begin(a), std::end(a), std::begin(array));
            return *this;
        }
        template <typename U>
        Array &operator=(const U (&a)[D]) {
            std::copy(std::begin(a), std::end(a), std::begin(array));
            return *this;
        }
        template <typename U>
        Array &operator=(const U &v) {
            std::fill(std::begin(array), std::end(array), v);
            return *this;
        }
        T *begin() { return &array[0]; }
        T *end() { return &array[D]; }
        const T *begin() const { return &array[0]; }
        const T *end() const { return &array[D]; }
    private:
        T array[D];
};

// Conversion.
template <typename T1, typename T2> struct ArrayDims;
template <typename T, size_t... Ds>
struct ArrayDims<T, Dims<Ds...>> {
    using type = Array<T, Ds...>;
};

/*
 * Base classes:  These are used as base classes.  HasInputLayer means that it is a layer that accepts input.
 * It does *not* mean that it has an InputLayer (which would be an alternative way to parse the term).
 * HasOutputLayer means that it is a layer that has output.
 */

template <typename T> class HasInputLayer;
template <typename T> class HasOutputLayer;

// Accepts input of the given dimensions.
template <size_t IN_D, size_t IN_H, size_t IN_W>
class HasInputLayer<Dims<IN_D, IN_H, IN_W>> {
    public:
        using InputDims = Dims<IN_D, IN_H, IN_W>;
        using Input = typename ArrayDims<Precision, InputDims>::type;
        HasInputLayer() : previous_layer(nullptr) {
            // Help debugging.
            downstream_deriv = std::numeric_limits<double>::signaling_NaN();
        }
        // Traing modifies and stores the output so that it can be used during backprop.  The last layer will
        // call backprop backward through the layers.
        virtual void train(const int label, const double minibatch_size) = 0;
        virtual void update_weights(const float rate) = 0;
        // Used for checking derivative numerically.
        virtual double loss(const Input &in, const int label) = 0;
        virtual int predict(const Input &) = 0;

    public:
        HasOutputLayer<InputDims> *previous_layer;
    protected:
        // This is passed to the previous layer during backprop.  However, it could be created as simply a
        // temporary array.  The only reason to keep it around is to check whether or not it has been computed
        // correctly.  In other words, it's for debugging.
        Input downstream_deriv;
};

template <typename T> class HasOutputLayer;
template <size_t OUT_D, size_t OUT_H, size_t OUT_W>
class HasOutputLayer<Dims<OUT_D, OUT_H, OUT_W>> {
    public:
        using OutputDims = Dims<OUT_D, OUT_H, OUT_W>;
        using Output = typename ArrayDims<Precision, OutputDims>::type;
        HasOutputLayer() : next_layer(nullptr) {}
        virtual void backprop(const Output &deriv, const double mb_size) = 0;
    public:
        HasInputLayer<OutputDims> *next_layer;
        // Leave public for now so that we can debug easily.
        Output output;
};

/*
 * This layer accepts an input image from MNIST.
 */

template <typename OUT_DIMS>
class InputLayer : public HasOutputLayer<OUT_DIMS> {
    public:
        using OutputIF = HasOutputLayer<OUT_DIMS>;
        using typename OutputIF::Output;
        constexpr static size_t OUT_D = OutputIF::OutputDims::D;
        constexpr static size_t OUT_H = OutputIF::OutputDims::H;
        constexpr static size_t OUT_W = OutputIF::OutputDims::W;
        static_assert(OUT_D == 1);
        static_assert(OUT_H >= 1);
        static_assert(OUT_W >= 1);
    public:
        // This is not virtual, because only layers that have input have train() as part of their interface.
        void train(const float (&image)[OUT_H][OUT_W], const int label, const double mb_size) {
            this->output[0] = image;
            this->next_layer->train(label, mb_size);
        }
        // Because it has output, this function must be defined, but there is no where to backprop to, so
        // there is no need for it to do anything.
        virtual void backprop(const Output &, const double) override { }
        // This is not virtual, because only layers that have input have update_weights() as part of their
        // interface.
        void update_weights(const float rate) {
            this->next_layer->update_weights(rate);
        }
        // This is not virtual, because only layers that have input have predict() as part of their interface.
        int predict(const float (&image)[OUT_H][OUT_W]) {
            Output output;
            output[0] = image;
            return this->next_layer->predict(output);
        }
};

/*
 * ConvolutionalLayer
 */

template <typename IN_DIMS, size_t N_FILTERS>
class ConvolutionalLayer : public HasInputLayer<IN_DIMS>,
 public HasOutputLayer<Dims<N_FILTERS, IN_DIMS::H, IN_DIMS::W>> {
        using InputIF = HasInputLayer<IN_DIMS>;
        using OutputIF = HasOutputLayer<Dims<N_FILTERS, IN_DIMS::H, IN_DIMS::W>>;
        using typename InputIF::Input;
        using typename OutputIF::Output;
        constexpr static size_t IN_D = InputIF::InputDims::D;
        constexpr static size_t IN_H = InputIF::InputDims::H;
        constexpr static size_t IN_W = InputIF::InputDims::W;
        constexpr static size_t OUT_D = OutputIF::OutputDims::D;
        constexpr static size_t OUT_H = OutputIF::OutputDims::H;
        constexpr static size_t OUT_W = OutputIF::OutputDims::W;
        static_assert(IN_D >= 1);
        static_assert(IN_H >= 1);
        static_assert(IN_W >= 1);
        static_assert(OUT_D >= 1);
        static_assert(OUT_H >= 1);
        static_assert(OUT_W >= 1);
        using Filter = Array<double, N_FILTERS, IN_D, FILTER_H, FILTER_W>;
        using Bias = Array<double, N_FILTERS>;

    public:

        // seed_seq is used to give each CL a different seed, so that we don't use the same initialization for
        // every layer.
        ConvolutionalLayer(const std::string &n, int seed_seq);
        // This layer has no loss function, so will always call it's forward layer.  If it has no forward
        // layer, that's a bug.
        virtual void train(const int label, const double mb_size) override {
            this->forward(this->previous_layer->output, m_filter, m_bias, this->output);
            assert(this->next_layer != nullptr);
            this->next_layer->train(label, mb_size);
        }
        virtual void update_weights(const float rate) override;
        virtual int predict(const Input &in) override {
            Output tmp_out;
            this->forward(in, m_filter, m_bias, tmp_out);
            return this->next_layer->predict(tmp_out);
        }
        virtual double loss(const Input &in, const int label) override {
            Output tmp_out;
            this->forward(in, m_filter, m_bias, tmp_out);
            return this->next_layer->loss(tmp_out, label);
        }
        virtual void backprop(const Output &upstream_deriv, const double mb_size) override;
        // Just saving this, not really needed.
        void backprop_old1(const Output &upstream_deriv, const double mb_size);
        // Do numerical check on weight derivative.
        void check_weight_derivative(const int label);
        // Do numerical check on downstream derivative.
        void check_downstream_derivative(const int label);

    private:

        // Do the forward computation.  Note that this does not use anything directly from the object, and so
        // is a static function.  The reason we don't use anything from the object is because we need to do
        // this for many different reasons, such as checking the derivative.  In such case, we don't want to
        // effect any change to the outputs, etc.
        static void forward(const Input &input, const Filter &filter, const Bias &bias, Output &output);

    public:

        const std::string m_name;
        Filter m_filter;
        Bias m_bias;
        Array<double, N_FILTERS, IN_D, FILTER_H, FILTER_W> m_filter_deriv;
        Array<double, N_FILTERS> m_bias_deriv;
};

template <typename IN_DIMS, size_t N_FILTERS>
ConvolutionalLayer<IN_DIMS, N_FILTERS>::ConvolutionalLayer(const std::string &n, int seed_seq) : m_name(n) {

    // seed_seq is used to give each CL a different seed, so that we don't use the same initialization for
    // every layer.
    std::default_random_engine eng(11'159'873 + seed_seq);
    std::normal_distribution<double> init;
    for (size_t g = 0; g < N_FILTERS; g++) {
        for (size_t h = 0; h < IN_D; h++) {
            for (size_t i = 0; i < FILTER_H; i++) {
                for (size_t j = 0; j < FILTER_W; j++) {
                    // Initialization picks Gaussian and divides by sqrt of "support" of each filter.
                    m_filter(g, h, i, j) = init(eng)/sqrt(IN_D*FILTER_H*FILTER_W);
                }
            }
        }
        m_bias(g) = 0;
    }

    // These need to initialized to zero, because it is the weight update operation that resets them
    // to 0.  During training, they accumulate, so that we can do minibatches.
    m_filter_deriv = 0;
    m_bias_deriv = 0;

    #if 0
    // Test code.
    double x = 0;
    for (size_t g = 0; g < N_FILTERS; g++) {
        for (size_t h = 0; h < IN_D; h++) {
            for (size_t i = 0; i < FILTER_H; i++) {
                for (size_t j = 0; j < FILTER_W; j++) {
                    m_filter(g, h, i, j) = ++x;
                }
            }
        }
        m_bias(g) = ++x;
    }
    m_filter_deriv = 0;
    m_bias_deriv = 0;
    #endif
}

template <typename IN_DIMS, size_t N_FILTERS>
void
ConvolutionalLayer<IN_DIMS, N_FILTERS>::update_weights(const float rate) {
    for (size_t f_g = 0; f_g < N_FILTERS; f_g++) {
        for (size_t f_h = 0; f_h < IN_D; f_h++) {
            for (size_t f_i = 0; f_i < FILTER_H; f_i++) {
                for (size_t f_j = 0; f_j < FILTER_W; f_j++) {
                    m_filter(f_g, f_h, f_i, f_j) += -rate*m_filter_deriv(f_g, f_h, f_i, f_j);
                    m_filter_deriv(f_g, f_h, f_i, f_j) = 0;
                }
            }
        }
        m_bias(f_g) += -rate*m_bias_deriv(f_g);
        m_bias_deriv(f_g) = 0;
    }
    this->next_layer->update_weights(rate);
}

template <typename IN_DIMS, size_t N_FILTERS>
void
ConvolutionalLayer<IN_DIMS, N_FILTERS>::backprop(const Output &upstream_deriv, const double mb_size) {

    /*
    std::cerr << "Upstream derivative:" << std::endl;
    std::cerr << upstream_deriv;
    */

    using ll_t = long long;

    this->downstream_deriv = 0;

    // Compute downstream derivatives.  Note that we slide over the output, not the input.  It can probably also
    // be done sliding over the input, but I think it would be significantly harder.
    auto &input(this->previous_layer->output);
    for (size_t out_i = 0; out_i < OUT_H; out_i++) {
        for (size_t out_j = 0; out_j < OUT_W; out_j++) {

            // We do another "convolution", but instead of computing the dot product, we compute the
            // derivatives.  We do this by essentially iterating over each pair of elements that would have been
            // multipled together for the dot product, but instead of computing the dot product, we add the
            // appropriate amount into the input and weight derivatives, using the product rule for computing
            // derivatives.

            // Compute the part of the filter that covers up actual input, not the padding.  We do not need to
            // go over the part that covers the padding.  While that part of the input does actually have a
            // derivative, because if it changed, it would in theory change the loss, we can't change it, so we
            // never need the derivative.  We also know that that part of the input cannot contribute to the
            // weight derivative, because the weight derivative is zero for that part (by the product rule for
            // computing derivatives).
            const size_t f_beg_i = std::max(0LL, -ll_t(out_i) + PADDING);
            const size_t f_beg_j = std::max(0LL, -ll_t(out_j) + PADDING);
            const size_t f_end_i = std::min(ll_t(FILTER_H), ll_t(OUT_H) + PADDING - ll_t(out_i));
            const size_t f_end_j = std::min(ll_t(FILTER_W), ll_t(OUT_W) + PADDING - ll_t(out_j));
            /*
            fprintf(stderr, "Filter index ranges: out(%zu, %zu), i(%zu, %zu], j(%zu, %zu]\n",
             out_i, out_j,
             f_beg_i, f_end_i,
             f_beg_j, f_end_j
            );
            */
            assert(f_beg_i < FILTER_H);
            assert(f_beg_j < FILTER_W);
            assert(f_end_i <= FILTER_H);
            assert(f_end_j <= FILTER_W);
            // Note we iterate over the filters, and over the entire input depth.
            for (size_t f_g = 0; f_g < N_FILTERS; f_g++) {
                // Since it is going through a ReLU, if the output is not greater than 0, then the
                // downstream and weight derivative is 0.
                if (this->output(f_g, out_i, out_j) > 0) {
                    for (size_t f_h = 0; f_h < IN_D; f_h++) {
                        for (size_t f_i = f_beg_i; f_i < f_end_i; f_i++) {
                            for (size_t f_j = f_beg_j; f_j < f_end_j; f_j++) {
                                // Note that the output layer depth index is the index of the filter, not the
                                // depth index of the filter.
                                const size_t in_i = out_i + f_i - PADDING;
                                const size_t in_j = out_j + f_j - PADDING;
                                this->downstream_deriv(f_h, in_i, in_j) +=
                                 m_filter(f_g, f_h, f_i, f_j)*upstream_deriv(f_g, out_i, out_j);
                                /*
                                fprintf(stderr, "layer %s, filter_deriv(%zu, %zu, %zu, %zu) added %f\n",
                                 m_name.c_str(), f_g, f_h, f_i, f_j, 
                                 input(f_h, in_i, in_j)*upstream_deriv(f_g, out_i, out_j)/mb_size);
                                */
                                m_filter_deriv(f_g, f_h, f_i, f_j) +=
                                 input(f_h, in_i, in_j)*upstream_deriv(f_g, out_i, out_j)/mb_size;
                            }
                        }
                    }
                    m_bias_deriv(f_g) += upstream_deriv(f_g, out_i, out_j)/mb_size;
                }
            }
        }
    }

    this->previous_layer->backprop(this->downstream_deriv, mb_size);
}

/*
 * This version computes the derivatives in a different way, by sliding the filter around each
 * pixel in the input.  It is more complicated, but I leave it here for reference.
 */

template <typename IN_DIMS, size_t N_FILTERS>
void
ConvolutionalLayer<IN_DIMS, N_FILTERS>::backprop_old1(const Output &upstream_deriv, const double mb_size) {

    std::cerr << "Upstream derivative:" << std::endl;
    std::cerr << upstream_deriv;

    using ll_t = long long;

    this->downstream_deriv = 0;

    // Compute downstream derivatives.
    for (size_t in_h = 0; in_h < IN_D; in_h++) {
        for (size_t in_i = 0; in_i < IN_H; in_i++) {
            for (size_t in_j = 0; in_j < IN_W; in_j++) {

                // Where does this input contribute to the loss?  Note that the convolution is in 2-D, but the
                // filter extends in 3-D.  So this is a bit tricky.  The depth index is the same as the depth
                // index of the input.  In other words, a given pixel in the input is only ever convolved at the
                // same depth in the filter, but changes x-y in the filter.

                // Compute where we start and end over the filter.  The start position can be figured out by
                // putting the filter as far to the right as it will go, and figure out, at the given image
                // position, where in the filter it is.  If before the beginning, then bump it up to 0.  The
                // same idea can be used to figure out the end position over the filter.
                const size_t f_beg_i = std::max(0LL, ll_t(FILTER_H) - (ll_t(IN_H) + PADDING - ll_t(in_i)));
                const size_t f_beg_j = std::max(0LL, ll_t(FILTER_W) - (ll_t(IN_W) + PADDING - ll_t(in_j)));
                const size_t f_end_i = std::min(ll_t(FILTER_H), ll_t(in_i) + PADDING + 1);
                const size_t f_end_j = std::min(ll_t(FILTER_W), ll_t(in_j) + PADDING + 1);
                fprintf(stderr, "Filter index ranges: in(%zu, %zu, %zu), i(%zu, %zu], j(%zu, %zu]\n",
                 in_h, in_i, in_j,
                 f_beg_i, f_end_i,
                 f_beg_j, f_end_j
                );
                assert(f_beg_i < FILTER_H);
                assert(f_beg_j < FILTER_W);
                assert(f_end_i <= FILTER_H);
                assert(f_end_j <= FILTER_W);
                for (size_t f_g = 0; f_g < N_FILTERS; f_g++) {
                    for (size_t f_i = f_beg_i; f_i < f_end_i; f_i++) {
                        for (size_t f_j = f_beg_j; f_j < f_end_j; f_j++) {
                            // Compute the output that this actually contributes to.
                            const size_t up_i = ll_t(in_i) + ll_t(PADDING) - ll_t(f_i);
                            const size_t up_j = ll_t(in_j) + ll_t(PADDING) - ll_t(f_j);
                            assert(up_i < OUT_H);
                            assert(up_j < OUT_W);
                            fprintf(stderr, "Downstream: (%zu, %zu, %zu), filter %zu (%zu, %zu, %zu), upstream(%zu, %zu, %zu)\n",
                             in_h, in_i, in_j,
                             f_g, in_h, f_i, f_j,
                             f_g, up_i, up_j);
                            // Since it is going through a ReLU, if the output is not greater than 0, then
                            // the downstream derivative is 0.
                            if (this->output(f_g, up_i, up_j) > 0) {
                                // Note that the output layer depth index is the index of the filter, not
                                // the depth index of the filter.
                                this->downstream_deriv(in_h, in_i, in_j)
                                 += m_filter(f_g, in_h, f_i, f_j)
                                 *upstream_deriv(f_g, up_i, up_j);
                            }
                        }
                    }
                }
            }
        }
    }

    auto &input(this->previous_layer->output);

    // Compute weight derivatives.
    for (size_t f_g = 0; f_g < N_FILTERS; f_g++) {
        for (size_t f_h = 0; f_h < IN_D; f_h++) {
            for (size_t f_i = 0; f_i < FILTER_H; f_i++) {
                for (size_t f_j = 0; f_j < FILTER_W; f_j++) {

                    // Where does this filter weight contribute to the loss?  Note that we don't need to iterate
                    // through the depth of the input, because a particular filter weight usage is fixed to the
                    // depth that it's at.  Also, we can just iterate through the actual input, not the padding,
                    // because the padding is set to 0, and so we know that the derivative there is 0.

                    // Best way to think about the computations below is that we are figuring out what parts of
                    // the input the filter weight covers as we slide it around.

                    // Compute how far to the left we begin the iteration through the image.  Put the filter all
                    // the way to left.  Find the position of the weight, in the image.  If it is off the left
                    // edge, then bump it up to 0.
                    const size_t in_i_begin = std::max(0LL, ll_t(f_i) - PADDING);
                    const size_t in_j_begin = std::max(0LL, ll_t(f_j) - PADDING);
                    // fprintf(stderr, "WD: in_begin(%zu, %zu)\n", in_i_begin, in_j_begin);
                    assert(in_i_begin < IN_H);
                    assert(in_j_begin < IN_W);
                    // Compute how far to the end we iterate.  Put the filter all the way to the end, so that
                    // the right edge is against the end of the padding.  Then, use position of the weight under
                    // consideration to limit how far to the right we can go.  Compute that by starting 2 pixels
                    // past the end.  Then we go back from that the number of pixels from the right end of the
                    // filter that we are at.  Then finally add one to be one past that.
                    fprintf(stderr, "%lld - %lld\n",
                     (ll_t(IN_W) + PADDING - 1),
                     (ll_t(FILTER_W) - (ll_t(f_j) - 1)));
                    const size_t in_i_end = std::min(ll_t(IN_H) - 1, (ll_t(IN_H) + PADDING - 1) - (ll_t(FILTER_H) - ll_t(f_i) - 1)) + 1;
                    const size_t in_j_end = std::min(ll_t(IN_W) - 1, (ll_t(IN_W) + PADDING - 1) - (ll_t(FILTER_W) - ll_t(f_j) - 1)) + 1;
                    // fprintf(stderr, "WD: in_end(%zu, %zu)\n", in_i_end, in_j_end);
                    assert(in_i_end <= IN_H);
                    assert(in_j_end <= IN_W);
                    fprintf(stderr, "WD: filter(%zu, %zu, %zu, %zu) range over i[%zu, %zu), j[%zu, %zu)\n",
                     f_g, f_h, f_i, f_j,
                     in_i_begin, in_i_end,
                     in_j_begin, in_j_end);
                    for (size_t in_i = in_i_begin; in_i < in_i_end; in_i++) {
                        for (size_t in_j = in_j_begin; in_j < in_j_end; in_j++) {
                            // Since it is going through a ReLU, if the output is not greater than 0, then
                            // the downstream derivative is 0.
                            if (this->output(f_g, in_i, in_j) > 0) {
                                m_filter_deriv(f_g, f_h, f_i, f_j)
                                 += input(f_h, in_i, in_j)*upstream_deriv(f_g, in_i, in_j)/mb_size;
                            }
                        }
                    }
                }
            }
        }

        // Compute bias derivative.
        for (size_t up_i = 0; up_i < OUT_H; up_i++) {
            for (size_t up_j = 0; up_j < OUT_W; up_j++) {
                // Since it is going through a ReLU, if the output is not greater than 0, then the downstream
                // derivative is 0.
                if (this->output(f_g, up_i, up_j) > 0) {
                    m_bias_deriv(f_g) += upstream_deriv(f_g, up_i, up_j)/mb_size;
                }
            }
        }
    }
}

template <typename IN_DIMS, size_t N_FILTERS>
void
ConvolutionalLayer<IN_DIMS, N_FILTERS>::check_weight_derivative(const int label) {

    fprintf(stderr, "Checking weight derivative of %s.\n", m_name.c_str());

    auto tmp_filter(m_filter);
    auto tmp_bias(m_bias);

    auto loss_weight = [&]() {
        Output temp_output;
        this->forward(this->previous_layer->output, tmp_filter, tmp_bias, temp_output);
        return this->next_layer->loss(temp_output, label);
    };

    for (size_t g = 0; g < N_FILTERS; g++) {

        /*
         * Check weight derivative.
         */

        for (size_t h = 0; h < IN_D; h++) {
            for (size_t i = 0; i < FILTER_H; i++) {
                for (size_t j = 0; j < FILTER_W; j++) {

                    double save = tmp_filter(g, h, i, j);

                    tmp_filter(g, h, i, j) = save - DELTA_H;
                    double Lminus = loss_weight();

                    tmp_filter(g, h, i, j) = save + DELTA_H;
                    double Lplus = loss_weight();

                    tmp_filter(g, h, i, j) = save;

                    double numeric_deriv = (Lplus - Lminus)/(2*DELTA_H);
                    double diff_deriv = m_filter_deriv(g, h, i, j);
                    if (derivative_error(numeric_deriv, diff_deriv) > 1E-6) {
                        fprintf(stderr, "WARNING: filter(%zu, %zu, %zu, %zu): numeric=%f, differentiated=%f\n", g, h, i, j,
                         numeric_deriv,
                         diff_deriv);
                    }
                }
            }
        }

        /*
         * Check bias derivative.
         */

        double save = tmp_bias(g);

        tmp_bias(g) = save - DELTA_H;
        double Lminus = loss_weight();

        tmp_bias(g) = save + DELTA_H;
        double Lplus = loss_weight();

        tmp_bias(g) = save;

        double numeric_deriv = (Lplus - Lminus)/(2*DELTA_H);
        double diff_deriv = m_bias_deriv(g);
        if (derivative_error(numeric_deriv, diff_deriv) > 1E-6) {
            fprintf(stderr, "filter(%zu) bias derivative: numeric=%f, differentiated=%f\n", g,
             numeric_deriv,
             diff_deriv);
        }
    }
}

template <typename IN_DIMS, size_t N_FILTERS>
void
ConvolutionalLayer<IN_DIMS, N_FILTERS>::check_downstream_derivative(const int label) {

    fprintf(stderr, "Checking downstream derivative of %s.\n", m_name.c_str());

    Input temp(this->previous_layer->output);

    for (size_t in_h = 0; in_h < IN_D; in_h++) {
        for (size_t in_i = 0; in_i < IN_H; in_i++) {
            for (size_t in_j = 0; in_j < IN_W; in_j++) {

                double save = temp[in_h][in_i][in_j];

                temp[in_h][in_i][in_j] = save - DELTA_H;
                double Lminus = loss(temp, label);

                temp[in_h][in_i][in_j] = save + DELTA_H;
                double Lplus = loss(temp, label);

                temp[in_h][in_i][in_j] = save;

                double numeric_deriv = (Lplus - Lminus)/(2*DELTA_H);
                double diff_deriv = this->downstream_deriv[in_h][in_i][in_j];
                if (derivative_error(numeric_deriv, diff_deriv) > 1E-6) {
                    fprintf(stderr, "%lu, %lu, %lu: numeric=%f, differentiated=%f\n", in_h, in_i, in_j,
                     numeric_deriv,
                     this->downstream_deriv[in_h][in_i][in_j]);
                }
            }
        }
    }
}

template <typename IN_DIMS, size_t N_FILTERS>
void
ConvolutionalLayer<IN_DIMS, N_FILTERS>::forward(const Input &input, const Filter &filter, const Bias &bias, Output &output) {

    Array<double, IN_D, IN_H + 2*PADDING, IN_W + 2*PADDING> in_padded;
    in_padded = std::numeric_limits<double>::signaling_NaN(); // For debugging.
    constexpr size_t in_padded_h = IN_H + 2*PADDING;
    constexpr size_t in_padded_w = IN_W + 2*PADDING;

    // First add padding.
    for (size_t h = 0; h < IN_D; h++) {
        // Add to sides of each row.
        for (size_t i = 0; i < in_padded_h; i++) {
            for (size_t p = 0; p < PADDING; p++) {
                in_padded(h, i, p) = 0;
                in_padded(h, i, in_padded_w - (PADDING - p)) = 0;
            }
        }
        // Add to top and bottom.
        for (size_t j = PADDING; j < in_padded_w - PADDING; j++) {
            for (size_t p = 0; p < PADDING; p++) {
                in_padded(h, p, j) = 0;
                in_padded(h, in_padded_h - (PADDING - p), j) = 0;
            }
        }
        // Copy middle.
        for (size_t i = PADDING; i < in_padded_h - PADDING; i++) {
            for (size_t j = PADDING; j < in_padded_w - PADDING; j++) {
                in_padded(h, i, j) = input(h, i - PADDING, j - PADDING);
            }
        }
    }

    for (size_t g = 0; g < N_FILTERS; g++) {
        for (size_t i = 0; i < OUT_H; i++) {
            for (size_t j = 0; j < OUT_W; j++) {
                double &out(output[g][i][j]);
                out = 0;
                for (size_t in_h = 0; in_h < IN_D; in_h++) {
                    for (size_t f_i = 0; f_i < FILTER_H; f_i++) {
                        for (size_t f_j = 0; f_j < FILTER_W; f_j++) {
                            /*
                            fprintf(stderr, "Added to DP: %f*%f\n",
                             double(filter(g, in_h, f_i, f_j)),
                             double(in_padded(in_h, i + f_i, j + f_j)));
                            */
                            out += filter(g, in_h, f_i, f_j)*in_padded(in_h, i + f_i, j + f_j);
                        }
                    }
                }
                // fprintf(stderr, "Bias to DP: %f\n", (double) bias[g]);
                out += bias[g];
                // ReLU
                out = std::max(0.0, out);
            }
        }
    }
}

/*
 * MaxPoolLayer
 */

template <typename IN_DIMS>
class MaxPoolLayer : public HasInputLayer<IN_DIMS>,
 public HasOutputLayer<Dims<IN_DIMS::D, IN_DIMS::H/2, IN_DIMS::W/2>> {

    public:

        using InputIF = HasInputLayer<IN_DIMS>;
        using OutputIF = HasOutputLayer<Dims<IN_DIMS::D, IN_DIMS::H/2, IN_DIMS::W/2>>;
        using typename InputIF::Input;
        using typename OutputIF::Output;
        constexpr static size_t IN_D = InputIF::InputDims::D;
        constexpr static size_t IN_H = InputIF::InputDims::H;
        constexpr static size_t IN_W = InputIF::InputDims::W;
        constexpr static size_t OUT_D = OutputIF::OutputDims::D;
        constexpr static size_t OUT_H = OutputIF::OutputDims::H;
        constexpr static size_t OUT_W = OutputIF::OutputDims::W;
        static_assert(IN_D >= 1);
        static_assert(IN_H >= 1);
        static_assert(IN_W >= 1);
        static_assert(OUT_D >= 1);
        static_assert(OUT_H >= 1);
        static_assert(OUT_W >= 1);

        MaxPoolLayer(const std::string &n) : m_name(n) {}

        // This layer has no loss function, so will always call it's forward layer.  If it has no forward
        // layer, that's a bug.
        virtual void train(const int label, const double mb_size) override {
            this->forward(this->previous_layer->output, this->output);
            this->next_layer->train(label, mb_size);
        }

        virtual void backprop(const Output &upstream_deriv, const double mb_size) override;

        virtual void update_weights(const float rate) override {
            // No weights/parameters in this layer.
            this->next_layer->update_weights(rate);
        }

        virtual double loss(const Input &in, const int label) override {
            Output temp_output;
            this->forward(in, temp_output);
            return this->next_layer->loss(temp_output, label);
        }

        virtual int predict(const Input &in) override {
            Output out;
            this->forward(in, out);
            return this->next_layer->predict(out);
        }

        void check_downstream_derivative(const int label);

    private:

        // This can't be static because it saves the indices of the max for use by the backward pass.
        void forward(const Input &input, Output &output);

    private:

        const std::string m_name;
        Array<size_t, OUT_D, OUT_H, OUT_W> m_max_index_i, m_max_index_j;
};

template <typename IN_DIMS>
void
MaxPoolLayer<IN_DIMS>::backprop(const Output &upstream_deriv, const double mb_size) {

    this->downstream_deriv = 0;

    for (size_t out_h = 0; out_h < OUT_D; out_h++) {
        for (size_t out_i = 0; out_i < OUT_H; out_i++) {
            for (size_t out_j = 0; out_j < OUT_W; out_j++) {
                this->downstream_deriv(out_h, m_max_index_i(out_h, out_i, out_j), m_max_index_j(out_h, out_i, out_j))
                 = upstream_deriv(out_h, out_i, out_j);
            }
        }
    }

    this->previous_layer->backprop(this->downstream_deriv, mb_size);
}

template <typename IN_DIMS>
void
MaxPoolLayer<IN_DIMS>::check_downstream_derivative(const int label) {

    fprintf(stderr, "Checking downstream derivative of %s.\n", m_name.c_str());

    Input temp(this->previous_layer->output);

    for (size_t in_h = 0; in_h < IN_D; in_h++) {
        for (size_t in_i = 0; in_i < IN_H; in_i++) {
            for (size_t in_j = 0; in_j < IN_W; in_j++) {

                double &input(temp(in_h, in_i, in_j));
                const double save = input;

                input = save - DELTA_H;
                const double Lminus = loss(temp, label);
                input = save + DELTA_H;
                const double Lplus = loss(temp, label);
                input = save;

                const double numeric_deriv = (Lplus - Lminus)/(2*DELTA_H);
                const double diff_deriv = this->downstream_deriv[in_h][in_i][in_j];
                if (derivative_error(numeric_deriv, diff_deriv) > 1E-6) {
                    fprintf(stderr, "WARNING: %lu, %lu, %lu: input=%f, numeric=%f, differentiated=%f\n", in_h, in_i, in_j,
                     input,
                     numeric_deriv,
                     diff_deriv);
                }
            }
        }
    }
}

template <typename IN_DIMS>
void
MaxPoolLayer<IN_DIMS>::forward(const Input &input, Output &output) {

    // This function works correctly only if dims are divisible by two.
    assert(IN_H%2 == 0);
    assert(IN_W%2 == 0);

    // Set all indices to maximum values to trigger range check failures during backprop if any were not set
    // correctly.
    m_max_index_i = std::numeric_limits<size_t>::max();
    m_max_index_j = std::numeric_limits<size_t>::max();

    // Iterate over each depth matrix independently.
    for (size_t in_h = 0; in_h < IN_D; in_h++) {
        for (size_t in_i = 0; in_i < IN_H; in_i += 2) {
            for (size_t in_j = 0; in_j < IN_W; in_j += 2) {
                double max = input(in_h, in_i, in_j);
                size_t max_i = in_i, max_j = in_j;
                // In theory could skip element (0, 0), but probably not much, if any,
                // performance gain.
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        const double &in(input(in_h, in_i + i, in_j + j));
                        if (in > max) {
                            max = in;
                            max_i = in_i + i;
                            max_j = in_j + j;
                        }
                    }
                }
                output(in_h, in_i/2, in_j/2) = max;
                m_max_index_i(in_h, in_i/2, in_j/2) = max_i;
                m_max_index_j(in_h, in_i/2, in_j/2) = max_j;
            }
        }
    }
}

/*
 * FullyConnectedLayer
 */

template <typename IN_DIMS, size_t N_NEURONS>
class FullyConnectedLayer : public HasInputLayer<IN_DIMS>, public HasOutputLayer<Dims<1, 1, N_NEURONS>> {

        using InputIF = HasInputLayer<IN_DIMS>;
        using OutputIF = HasOutputLayer<Dims<1, 1, N_NEURONS>>;
        using typename InputIF::Input;
        using typename OutputIF::Output;
        constexpr static size_t IN_D = InputIF::InputDims::D;
        constexpr static size_t IN_H = InputIF::InputDims::H;
        constexpr static size_t IN_W = InputIF::InputDims::W;
        constexpr static size_t OUT_D = OutputIF::OutputDims::D;
        constexpr static size_t OUT_H = OutputIF::OutputDims::H;
        constexpr static size_t OUT_W = OutputIF::OutputDims::W;
        static_assert(OUT_D == 1);
        static_assert(OUT_H == 1);

    public:

        FullyConnectedLayer(const std::string &n, const bool relu, const double do_rate, const int seed_seq);
        // This layer has no loss function, so will always call it's forward
        // layer.  If it has no forward layer, that's a bug.
        virtual void train(const int label, const double mb_size) override {

            std::uniform_real_distribution<double> dist(0, 1);
            
            // Fill dropped array with either 0 if dropped, or 1/dropout_rate if not dropped, so that the
            // expected value of the output is constant.
            std::generate(m_current_kept.begin(), m_current_kept.end(),
             [&]() { return dist(m_eng) < m_keep_prob ? 1/m_keep_prob : 0; });
            
            this->forward(this->previous_layer->output, this->m_weight, this->m_bias, this->m_current_kept, this->output);
            this->next_layer->train(label, mb_size);
        }
        virtual void backprop(const Output &full_upstream_deriv, const double mb_size) override;
        virtual void update_weights(const float rate) override;
        void check_weight_derivative(const int label);
        void check_downstream_derivative(const int label);
        virtual double loss(const Input &in, const int label) override {
            Output temp_output;
            this->forward(in, this->m_weight, this->m_bias, this->m_all_kept, temp_output);
            return this->next_layer->loss(temp_output, label);
        }
        virtual int predict(const Input &in) override {
            Output out;
            this->forward(in, this->m_weight, this->m_bias, this->m_all_kept, out);
            return this->next_layer->predict(out);
        }

    private:

        // Could not make this static bcause it needed the m_relu flag.
        void forward(const Input &input, const Array<Input, N_NEURONS> &weight, const Array<double, N_NEURONS> &bias, const Array<double, N_NEURONS> &dropped, Output &output);

    public:

        const std::string m_name;
        const bool m_relu;
        Array<Input, N_NEURONS> m_weight;
        Array<Input, N_NEURONS> m_weight_deriv;
        Array<double, N_NEURONS> m_bias;
        Array<double, N_NEURONS> m_bias_deriv;
        const double m_keep_prob;
        Array<double, N_NEURONS> m_current_kept;
        const Array<double, N_NEURONS> m_all_kept;

        std::default_random_engine m_eng;
};

template <typename IN_DIMS, size_t N_NEURONS>
FullyConnectedLayer<IN_DIMS, N_NEURONS>::FullyConnectedLayer(const std::string &n, const bool relu, const double do_rate, const int seed_seq)
 : m_name(n), m_relu(relu), m_keep_prob(1 - do_rate), m_all_kept(1), m_eng(7389 + seed_seq) {

    std::normal_distribution<double> init;
    // For each neuron, plane, row, and colum...
    for (auto &n : m_weight) {
        for (auto &p : n) {
            for (auto &r : p) {
                for (auto &c : r) {
                    c = init(m_eng)/sqrt(IN_DIMS::N);
                }
            }
        }
    }

    m_bias = 0;

    m_weight_deriv = 0;
    m_bias_deriv = 0;
}

template <typename IN_DIMS, size_t N_NEURONS>
void
FullyConnectedLayer<IN_DIMS, N_NEURONS>::backprop(const Output &full_upstream_deriv, const double mb_size) {

    auto &upstream_deriv(full_upstream_deriv[0][0]);
    this->downstream_deriv = 0;
    auto &input(this->previous_layer->output);

    for (size_t i = 0; i < N_NEURONS; i++) {
        if (m_current_kept(i) > 0) {
            if (!m_relu || this->output(0, 0, i) > 0) {
                for (size_t in_h = 0; in_h < IN_D; in_h++) {
                    for (size_t in_i = 0; in_i < IN_H; in_i++) {
                        for (size_t in_j = 0; in_j < IN_W; in_j++) {
                            this->downstream_deriv[in_h][in_i][in_j] += m_current_kept(i)*upstream_deriv[i]*m_weight[i][in_h][in_i][in_j];
                            /*
                            fprintf(stderr, "%lu, %lu, %lu: %f\n",
                             in_h, in_i, in_j,
                             this->downstream_deriv[in_h][in_i][in_j]);
                            */
                            // Divide by minibatch size to get the average.
                            m_weight_deriv[i][in_h][in_i][in_j] += (m_current_kept(i)*upstream_deriv[i]*input[in_h][in_i][in_j])/mb_size;
                        }
                    }
                }
                m_bias_deriv(i) += (m_current_kept(i)*upstream_deriv[i])/mb_size;
            }
        }
    }
    this->previous_layer->backprop(this->downstream_deriv, mb_size);
}

template <typename IN_DIMS, size_t N_NEURONS>
void
FullyConnectedLayer<IN_DIMS, N_NEURONS>::update_weights(const float rate) {

    for (size_t i = 0; i < N_NEURONS; i++) {
        for (size_t in_h = 0; in_h < IN_D; in_h++) {
            for (size_t in_i = 0; in_i < IN_H; in_i++) {
                for (size_t in_j = 0; in_j < IN_W; in_j++) {
                     m_weight[i](in_h, in_i, in_j) -= rate*m_weight_deriv[i](in_h, in_i, in_j);
                     m_weight_deriv[i](in_h, in_i, in_j) = 0;
                }
            }
        }
        m_bias(i) -= rate*m_bias_deriv(i);
        m_bias_deriv(i) = 0;
    }

    this->next_layer->update_weights(rate);
}

template <typename IN_DIMS, size_t N_NEURONS>
void
FullyConnectedLayer<IN_DIMS, N_NEURONS>::check_weight_derivative(const int label) {

    fprintf(stderr, "Checking weight derivative of %s.\n", m_name.c_str());

    auto tmp_weight(m_weight);
    auto tmp_bias(m_bias);

    auto loss_weight = [&]() {
        Output temp_output;
        this->forward(this->previous_layer->output, tmp_weight, tmp_bias, temp_output);
        return this->next_layer->loss(temp_output, label);
    };

    for (size_t n = 0; n < N_NEURONS; n++) {

        /*
         * Check weight derivative.
         */

        for (size_t h = 0; h < IN_D; h++) {
            for (size_t i = 0; i < IN_H; i++) {
                for (size_t j = 0; j < IN_W; j++) {

                    double &w(tmp_weight(n)(h, i, j));
                    const double save = w;

                    w = save - DELTA_H;
                    const double Lminus = loss_weight();
                    w = save + DELTA_H;
                    const double Lplus = loss_weight();
                    w = save;

                    const double numeric_deriv = (Lplus - Lminus)/(2*DELTA_H);
                    const double diff_deriv = m_weight_deriv(n)(h, i, j);
                    if (derivative_error(numeric_deriv, diff_deriv) > 1E-6) {
                        fprintf(stderr, "WARNING: FC weight(%zu, %zu, %zu, %zu): numeric=%f, differentiated=%f\n", n, h, i, j,
                         numeric_deriv,
                         diff_deriv);
                    }
                }
            }
        }

        /*
         * Check bias derivative.
         */

        const double save = tmp_bias(n);

        tmp_bias(n) = save - DELTA_H;
        const double Lminus = loss_weight();
        tmp_bias(n) = save + DELTA_H;
        const double Lplus = loss_weight();
        tmp_bias(n) = save;

        const double numeric_deriv = (Lplus - Lminus)/(2*DELTA_H);
        const double diff_deriv =  m_bias_deriv(n);
        if (derivative_error(numeric_deriv, diff_deriv) > 1E-6) {
            fprintf(stderr, "WARNING: FC bias(%zu) bias derivative: numeric=%f, differentiated=%f\n", n,
             numeric_deriv,
             diff_deriv);
        }
    }
}

template <typename IN_DIMS, size_t N_NEURONS>
void
FullyConnectedLayer<IN_DIMS, N_NEURONS>::check_downstream_derivative(const int label) {

    fprintf(stderr, "Checking downstream derivative of %s.\n", m_name.c_str());

    Input temp(this->previous_layer->output);

    for (size_t in_h = 0; in_h < IN_D; in_h++) {
        for (size_t in_i = 0; in_i < IN_H; in_i++) {
            for (size_t in_j = 0; in_j < IN_W; in_j++) {

                double &input(temp(in_h, in_i, in_j));
                const double save = input;

                input = save - DELTA_H;
                const double Lminus = loss(temp, label);
                input = save + DELTA_H;
                const double Lplus = loss(temp, label);
                input = save;

                const double numeric_deriv = (Lplus - Lminus)/(2*DELTA_H);
                const double diff_deriv = this->downstream_deriv[in_h][in_i][in_j];
                if (derivative_error(numeric_deriv, diff_deriv) > 1E-6) {
                    fprintf(stderr, "%lu, %lu, %lu: numeric=%f, differentiated=%f\n", in_h, in_i, in_j,
                     numeric_deriv,
                     diff_deriv);
                }
            }
        }
    }
}

template <typename IN_DIMS, size_t N_NEURONS>
void
FullyConnectedLayer<IN_DIMS, N_NEURONS>::forward(const Input &input, const Array<Input, N_NEURONS> &weight, const Array<double, N_NEURONS> &bias,
 const Array<double, N_NEURONS> &dropped, Output &output) {
    // Connect each neuron to everything.
    for (size_t i = 0; i < N_NEURONS; i++) {
        double &out(output[0][0][i]);
        out = 0;
        for (size_t in_h = 0; in_h < IN_D; in_h++) {
            for (size_t in_i = 0; in_i < IN_H; in_i++) {
                for (size_t in_j = 0; in_j < IN_W; in_j++) {
                    out += weight[i][in_h][in_i][in_j]*input[in_h][in_i][in_j];
                }
            }
        }
        out += bias(i);
        if (m_relu) {
            out = std::max(0.0, out);
        }
        // Value is 0 if dropped, or 1/dropout-rate if not dropped, so as to maintain constant overall
        // expected value.
        assert(dropped(i) == 0 || dropped(i) >= 1);
        /*
        if (dropped(i) == 0) {
            fprintf(stderr, "%d dropped\n", int(i));
        } else if (dropped(i) > 1) {
            fprintf(stderr, "%d expanded by %f\n", int(i), dropped(i));
        }
        */
        out *= dropped(i);
    }
}

/*
 * SoftmaxLayer
 */

template <size_t N>
class SoftmaxLayer : public HasInputLayer<Dims<1, 1, N>>, public HasOutputLayer<Dims<1, 1, N>> {

        using InputIF = HasInputLayer<Dims<1, 1, N>>;
        using OutputIF = HasOutputLayer<Dims<1, 1, N>>;
        using typename InputIF::Input;
        using typename OutputIF::Output;

    public:

        // This layer has no loss function, so will always call it's forward layer.
        // If it has no forward layer, that's a bug.
        virtual void train(const int label, const double mb_size) override {
            forward(this->previous_layer->output, this->output);
            this->next_layer->train(label, mb_size);
        }

        virtual void backprop(const typename OutputIF::Output &full_upstream_deriv, const double mb_size) override;
        virtual void update_weights(const float rate) override {
            // No weights in this layer.
            this->next_layer->update_weights(rate);
        }
        virtual double loss(const Input &in, const int label) override {
            Output temp_output;
            this->forward(in, temp_output);
            return this->next_layer->loss(temp_output, label);
        }
        virtual int predict(const Input &in) override {
            /*
            std::cerr << "Predicting for: " << std::endl;
            for (auto x : in[0][0]) {
                std::cerr << x << std::endl;
            }
            std::cerr << std::endl;
            */
            auto pos = std::max_element(std::begin(in[0][0]), std::end(in[0][0]));
            return std::distance(std::begin(in[0][0]), pos);
        }

    private:

        static void forward(const Input &input, Output &output);
};

template <size_t N>
void
SoftmaxLayer<N>::backprop(const typename OutputIF::Output &full_upstream_deriv, const double mb_size) {

    // Note that we assume that ultimately we are computing the derivative of a scalar with respect to
    // each element of the softmax, so we simply add the derivatives.
    //
    auto &upstream_deriv(full_upstream_deriv[0][0]);
    this->downstream_deriv = 0;
    auto &downstream_deriv(this->downstream_deriv[0][0]);
    auto &output(this->output[0][0]);
    for (size_t j = 0; j < N; j++) {
        downstream_deriv[j] = 0;
        for (size_t i = 0; i < N; i++) {
            if (i == j) {
                downstream_deriv[j] += upstream_deriv[i]*(output[i]*(1 - output[j]));
            } else {
                downstream_deriv[j] += upstream_deriv[i]*(-output[j]*output[i]);
            }
        }
    }
    this->previous_layer->backprop(this->downstream_deriv, mb_size);
}

template <size_t N>
void
SoftmaxLayer<N>::forward(const Input &input, Output &output) {
    // Assume just a 1-D vector.  Note that this is a bit confusing,
    // because in C++, we think of this as just a single row, but
    // mathematically, we like to think of it as a column vector.
    auto &out(output[0][0]);
    auto &in(input[0][0]);
    // D is constant to improve numeric stability.
    const double D = *std::max_element(std::begin(in), std::end(in));
    double sum = 0;
    for (size_t i = 0; i < N; i++) {
        out[i] = exp(in[i] - D);
        sum += out[i];
    }
    for (size_t i = 0; i < N; i++) {
        out[i] = out[i]/sum;
    }
}

/*
 * CrossEntropyLayer
 */

template <size_t N>
class CrossEntropyLayer : public HasInputLayer<Dims<1, 1, N>> {
        using InputIF = HasInputLayer<Dims<1, 1, N>>;
        using typename InputIF::Input;
    public:
        virtual void train(const int label, const double mb_size) override {
            // Note that there is no actual need to calculate the loss at this point.
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wunused-variable"
            double loss = -log(this->previous_layer->output[0][0][label]);
            #pragma GCC diagnostic pop
            // fprintf(stderr, "loss: %f\n", loss);
            Input deriv;
            deriv = 0;
            this->downstream_deriv = 0;
            this->downstream_deriv[0][0][label] = -1/(this->previous_layer->output[0][0][label]);
            this->previous_layer->backprop(this->downstream_deriv, mb_size);
        }
        virtual void update_weights(const float) override {
            // No weights in this layer, and this layer has no output.
        }
        virtual double loss(const Input &in, const int label) override {
            return -std::log(in[0][0][label]);
        }
        virtual int predict(const Input &) override {
            assert(false);
            return -1;
        }
};

void
swap(int &i) {
    // Some of the & are superfluous.
    i =
     (0xff&(i >> 24)) |
     (0xff00&(i >> 8)) |
     (0xff0000&(i << 8)) |
     (0xff000000&(i << 24));
}

int
read_int(int fd) {
    int rv;
    int i;
    rv = read(fd, &i, 4); assert(rv == 4);
    swap(i);
    return i;
}

void
output_pgm(const std::string &fn, const float (&img)[28][28]) {

    std::ofstream ofs(fn, std::fstream::out|std::fstream::trunc);

    ofs << "P2\n";
    ofs << "28 28\n";
    ofs << "255\n";
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            if (j > 0) {
                ofs << " ";
            }
            ofs << 255 - int(std::round(127.5*(img[i][j] + 1)));
        }
        ofs << "\n";
    }
}

void
output_pgm(const std::string &fn, const Array<double, 1, 5, 5> &img) {

    std::ofstream ofs(fn, std::fstream::out|std::fstream::trunc);

    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            min = std::min(min, img(0, i, j));
            max = std::max(max, img(0, i, j));
        }
    }
    const double range = max - min;

    ofs << "P2\n";
    ofs << "5 5\n";
    ofs << "255\n";
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (j > 0) {
                ofs << " ";
            }
            ofs << int(std::round((img(0, i, j) - min)*(255/range)));
        }
        ofs << "\n";
    }
}

template <int N>
void
read_mnist_images(const std::string &fn, float (&imgs)[N][28][28]) {

    int rv;

    int fd;
    fd = open(fn.c_str(), O_RDONLY);
    assert(fd >= 0);

    int magic = read_int(fd);
    assert(magic == 0x803);

    int n_images = read_int(fd);
    assert(n_images == N);

    int n_rows = read_int(fd);
    assert(n_rows == 28);

    int n_cols = read_int(fd);
    assert(n_cols == 28);

    for (int i = 0; i < N; i++) {
        unsigned char tmp[28][28];
        rv = read(fd, tmp, 28*28); assert(rv == 28*28);
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                // Make go from -1 to 1.
                imgs[i][r][c] = double(tmp[r][c])/127.5 - 1;
            }
        }
    }

    rv = close(fd); assert(rv == 0);
}

template <int N>
void
read_mnist_labels(const std::string &fn, unsigned char (&labels)[N]) {

    int rv;

    int fd;
    fd = open(fn.c_str(), O_RDONLY);
    assert(fd >= 0);

    int magic = read_int(fd);
    assert(magic == 0x801);

    int n_labels = read_int(fd);
    assert(n_labels == N);

    rv = read(fd, labels, N); assert(rv == N);
    for (int i = 0; i < N; i++) {
        assert(labels[i] >= 0 && labels[i] <= 9);
    }

    rv = close(fd); assert(rv == 0);
}

void
run3() {

    /*
     * Read in training data.
     */

    static float training_images[60'000][28][28];
    read_mnist_images("mnist/train-images-idx3-ubyte", training_images);
    output_pgm("img0.pgm", training_images[0]);
    output_pgm("img59999.pgm", training_images[59999]);

    static unsigned char training_labels[60'000];
    read_mnist_labels("mnist/train-labels-idx1-ubyte", training_labels);
    assert(training_labels[0] == 5);
    assert(training_labels[59'999] == 8);

    static float test_images[10'000][28][28];
    read_mnist_images("mnist/t10k-images-idx3-ubyte", test_images);
    static unsigned char test_labels[10'000];
    read_mnist_labels("mnist/t10k-labels-idx1-ubyte", test_labels);

    {
        static InputLayer<Dims<1, 28, 28>> il;
        static ConvolutionalLayer<Dims<1, 28, 28>, 32> cl1("cl1", 1);
        static MaxPoolLayer<Dims<32, 28, 28>> pl1("pl1");
        static ConvolutionalLayer<Dims<32, 14, 14>, 32> cl2("cl2", 2);
        static MaxPoolLayer<Dims<32, 14, 14>> pl2("pl2");
        /*
        static FullyConnectedLayer<Dims<64, 7, 7>, 1024> dl1("dl1", 0.4, 1);
        static FullyConnectedLayer<Dims<1, 1, 1024>, 10> dl2("dl2", 0, 2);
        */
        static FullyConnectedLayer<Dims<32, 7, 7>, 64> dl1("dl1", true, 0.4, 1);
        static FullyConnectedLayer<Dims<1, 1, 64>, 10> dl2("dl2", false, 0, 2);
        static SoftmaxLayer<10> sm;
        static CrossEntropyLayer<10> ce;

        il.next_layer = &cl1; cl1.previous_layer = &il;
        cl1.next_layer = &pl1; pl1.previous_layer = &cl1;
        pl1.next_layer = &cl2; cl2.previous_layer = &pl1;
        cl2.next_layer = &pl2; pl2.previous_layer = &cl2;
        pl2.next_layer = &dl1; dl1.previous_layer = &pl2;
        dl1.next_layer = &dl2; dl2.previous_layer = &dl1;
        dl2.next_layer = &sm; sm.previous_layer = &dl2;
        sm.next_layer = &ce; ce.previous_layer = &sm;

        std::mt19937 g(9815);
        std::uniform_int_distribution<size_t> pick_test(0, 9'999);

        for (int e = 0; e < 100; e++) {

            // Create shuffled sequence of training images.
            std::vector<int> training(60'000);
            std::iota(training.begin(), training.end(), 0);
            assert(*--training.end() == 59'999);
            std::shuffle(training.begin(), training.end(), g);

            /*
            // Create shuffled sequence of test images.
            std::vector<int> test(10'000);
            std::iota(test.begin(), test.end(), 0);
            assert(*--test.end() == 9'999);
            {
                std::mt19937 g(9175);
                std::shuffle(test.begin(), test.end(), g);
            }
            */

            // size_t training_index = 0;

            for (int r = 0; r < 600; r++) {

                if (r%50 == 0) {

                    // fprintf(stderr, "Begin predict...."); fflush(stderr);
                    int correct = 0;
                    for (size_t i = 0; i < 100; i++) {
                        // fprintf(stderr, "Predict: %d for %lu\n", input.predict(training_images[i]), i);
                        size_t ind = pick_test(g);
                        if (il.predict(test_images[ind]) == test_labels[ind]) {
                            correct++;
                        }
                    }
                    fprintf(stderr, "Epoch %d: Round %d: accuracy=%f\n", e, r, correct/100.0);

                    for (size_t i = 0; i < 32; i++) {
                        char buf[100];
                        sprintf(buf, "e%03d-r%03d-%02zu.pgm", e, r, i);
                        output_pgm(buf, cl1.m_filter[i]);
                    }
                    // std::cerr << cl1.m_filter;
                }

                /*
                std::cerr << "Weights:" << std::endl;
                for (size_t n = 0; n < 10; n++) {
                    std::cerr << "Neuron " << n << ":" << std::endl;
                    print(std::cout, dl.weight[n]);
                }
                */

                for (size_t i = 0; i < 100; i++) {
                    il.train(training_images[training.at(100*r + i)], training_labels[training.at(100*r + i)], 100);
                }
                il.update_weights(.01);
            }
        }
    }
}

// Two FC layers, with 1024 neurons in first.
void
run4() {
    static float training_images[60'000][28][28];
    read_mnist_images("mnist/train-images-idx3-ubyte", training_images);
    output_pgm("img0.pgm", training_images[0]);
    output_pgm("img59999.pgm", training_images[59999]);

    static unsigned char training_labels[60'000];
    read_mnist_labels("mnist/train-labels-idx1-ubyte", training_labels);
    assert(training_labels[0] == 5);
    assert(training_labels[59'999] == 8);

    static float test_images[10'000][28][28];
    read_mnist_images("mnist/t10k-images-idx3-ubyte", test_images);
    static unsigned char test_labels[10'000];
    read_mnist_labels("mnist/t10k-labels-idx1-ubyte", test_labels);

    static InputLayer<Dims<1, 28, 28>> il;
    static FullyConnectedLayer<Dims<1, 28, 28>, 1024> dl1("dl1", true, 0.4, 1);
    static FullyConnectedLayer<Dims<1, 1, 1024>, 10> dl2("dl2", false, 0.0, 2);
    static SoftmaxLayer<10> sm;
    static CrossEntropyLayer<10> ce;

    il.next_layer = &dl1; dl1.previous_layer = &il;
    dl1.next_layer = &dl2; dl2.previous_layer = &dl1;
    dl2.next_layer = &sm; sm.previous_layer = &dl2;
    sm.next_layer = &ce; ce.previous_layer = &sm;

    std::default_random_engine eng(9815);
    std::uniform_int_distribution<size_t> pick_test(0, 9'999);

    for (int e = 0; e < 2; e++) {
        // Create shuffled sequence of training images.
        std::vector<int> training(60'000);
        std::iota(training.begin(), training.end(), 0);
        assert(*--training.end() == 59'999);
        std::shuffle(training.begin(), training.end(), eng);

        for (int r = 0; r < 600; r++) {

            if (r%100 == 0) {
                // fprintf(stderr, "Begin predict...."); fflush(stderr);
                int correct = 0;
                for (size_t i = 0; i < 10'000; i++) {
                    // fprintf(stderr, "Predict: %d for %lu\n", il.predict(training_images[i]), i);
                    size_t ind = pick_test(eng);
                    if (il.predict(test_images[ind]) == test_labels[ind]) {
                        correct++;
                    }
                }
                fprintf(stderr, "Epoch %d: Round %d: accuracy=%f\n", e, r, correct/10'000.0);
            }

            /*
            std::cerr << "Weights:" << std::endl;
            for (size_t n = 0; n < 10; n++) {
                std::cerr << "Neuron " << n << ":" << std::endl;
                print(std::cout, dl.weight[n]);
            }
            */

            // Make faster -- Guaranteed 100 entries, run on 10x10 cuda cores
            // If in kernel, train has to be device function and all func that
            // train calls
            // __device function
            for (size_t i = 0; i < 100; i++) {
                il.train(training_images[training.at(100*r + i)], training_labels[training.at(100*r + i)], 100);
            }
            il.update_weights(.002);
        }
    }

    #if 0
    for (int r = 0; r < 100'000; r++) {

        if (r % 100 == 0) {

            int correct = 0;
            for (size_t i = 0; i < 100; i++) {
                // fprintf(stderr, "Predict: %d for %lu\n", input.predict(training_images[i]), i);
                if (input.predict(training_images[i]) == training_labels[i]) {
                    correct++;
                }
            }
            printf("Round %d: accuracy=%f\n", r, correct/100.0);
        }

        /*
        std::cerr << "Weights:" << std::endl;
        for (size_t n = 0; n < 10; n++) {
            std::cerr << "Neuron " << n << ":" << std::endl;
            print(std::cout, dl.weight[n]);
        }
        */

        for (size_t i = 0; i < 100; i++) {
            input.train(training_images[i], training_labels[i], 100);
        }
        input.update_weights(.001);
    }
    #endif
}

void run2();

int
main() {

    /*
     * Test Array.
     */
    {
        Array<double, 3> a;
        a[0] = 1.1;
        a[1] = 2.2;
        a[2] = 3.3;
        Array<double, 3> a2(a);
        assert(a2[0] == 1.1);
        assert(a2[1] == 2.2);
        assert(a2[2] == 3.3);
        Array<double, 3> a3;
        a3 = a2;
        assert(a3[0] == 1.1);
        assert(a3[1] == 2.2);
        assert(a3[2] == 3.3);
        // a3[3] = 1; // Should assert().
        Array<double, 2, 2> a4;
        a4(0, 0) = 0;
        a4(0, 1) = 1;
        a4(1, 0) = 2;
        a4(1, 1) = 3;
        assert(a4[0][0] == 0);
        assert(a4[0][1] == 1);
        assert(a4[1][0] == 2);
        assert(a4[1][1] == 3);
        Array<float, 2, 3> a5;
        a5 = 1.1f;
        assert(a5(0, 0) == 1.1f);
        assert(a5(0, 1) == 1.1f);
        assert(a5(0, 2) == 1.1f);
        assert(a5(1, 0) == 1.1f);
        assert(a5(1, 1) == 1.1f);
        assert(a5(1, 2) == 1.1f);
    }

    run4();
}

/*
 * Everything below here is unused.
 */

template <typename T, size_t D, size_t... Ds>
struct NativeArray {
    using type = typename NativeArray<T, Ds...>::type [D];
};
template <typename T, size_t D>
struct NativeArray<T, D> {
    using type = T [D];
};

void
test1() {

    InputLayer<Dims<1, 3, 3>> input;
    ConvolutionalLayer<Dims<1, 3, 3>, 2> cl1("cl1", 1);
    FullyConnectedLayer<Dims<2, 3, 3>, 2> dl("dl", false, 0, 1);
    SoftmaxLayer<2> sm;
    CrossEntropyLayer<2> ce;

    input.next_layer = &cl1; cl1.previous_layer = &input;
    cl1.next_layer = &dl; dl.previous_layer = &cl1;
    dl.next_layer = &sm; sm.previous_layer = &dl;
    sm.next_layer = &ce; ce.previous_layer = &sm;

    float image[3][3];
    image[0][0] = 1.1;
    image[0][1] = 1.2;
    image[0][2] = 1.3;
    image[1][0] = 2.1;
    image[1][1] = 2.2;
    image[1][2] = 2.3;
    image[2][0] = 3.1;
    image[2][1] = 3.2;
    image[2][2] = 3.3;
    input.train(image, 0, 1);

    std::cerr << "Filters: " << std::endl;
    std::cerr << cl1.m_filter;

    std::cerr << "Filter output: " << std::endl;
    std::cerr << cl1.output;

    std::cerr << "Weight derivatives: " << std::endl;
    std::cerr << cl1.m_filter_deriv;

    cl1.check_weight_derivative(0);

    std::cerr << "FC downstream deriv:" << std::endl;
    dl.check_downstream_derivative(0);
}

void
test2() {

    auto &input(*new InputLayer<Dims<1, 2, 2>>);
    auto &dl(*new FullyConnectedLayer<Dims<1, 2, 2>, 4>("dl", false, 0, 1));
    auto &sm(*new SoftmaxLayer<4>);
    auto &ce(*new CrossEntropyLayer<4>);

    input.next_layer = &dl; dl.previous_layer = &input;
    dl.next_layer = &sm; sm.previous_layer = &dl;
    sm.next_layer = &ce; ce.previous_layer = &sm;

    std::default_random_engine eng(691);
    std::uniform_real_distribution<float> dist(-1, 1);
    float img[2][2];
    for (auto &r : img) {
        for (auto &c : r) {
            c = dist(eng);
            // c = .1;
        }
    }

    input.train(img, 0, 1);

    dl.check_downstream_derivative(0);

    std::cerr << "Inputs:" << std::endl;
    std::cerr << input.output;

    std::cerr << "Weights:" << std::endl;
    for (size_t n = 0; n < 2; n++) {
        std::cerr << "Neuron" << n << ":" << std::endl;
        std::cerr << dl.m_weight[n];
    }
}

void
test3() {

    /*
     * Read in training data.
     */

    static float training_images[60'000][28][28];
    read_mnist_images("mnist/train-images-idx3-ubyte", training_images);
    output_pgm("img0.pgm", training_images[0]);
    output_pgm("img59999.pgm", training_images[59999]);

    static unsigned char training_labels[60'000];
    read_mnist_labels("mnist/train-labels-idx1-ubyte", training_labels);
    assert(training_labels[0] == 5);
    assert(training_labels[59'999] == 8);

    {
        static InputLayer<Dims<1, 28, 28>> il;
        static ConvolutionalLayer<Dims<1, 28, 28>, 16> cl1("cl1", 1);
        static MaxPoolLayer<Dims<16, 28, 28>> pl1("pl1");
        static ConvolutionalLayer<Dims<16, 14, 14>, 16> cl2("cl2", 2);
        static MaxPoolLayer<Dims<16, 14, 14>> pl2("pl2");
        static FullyConnectedLayer<Dims<16, 7, 7>, 32> dl1("dl1", true, 0, 1);
        static FullyConnectedLayer<Dims<1, 1, 32>, 10> dl2("dl2", false, 0, 2);
        static SoftmaxLayer<10> sm;
        static CrossEntropyLayer<10> ce;

        il.next_layer = &cl1; cl1.previous_layer = &il;
        cl1.next_layer = &pl1; pl1.previous_layer = &cl1;
        pl1.next_layer = &cl2; cl2.previous_layer = &pl1;
        cl2.next_layer = &pl2; pl2.previous_layer = &cl2;
        pl2.next_layer = &dl1; dl1.previous_layer = &pl2;
        dl1.next_layer = &dl2; dl2.previous_layer = &dl1;
        dl2.next_layer = &sm; sm.previous_layer = &dl2;
        sm.next_layer = &ce; ce.previous_layer = &sm;

        il.train(training_images[0], training_labels[0], 1);

        /*
        fprintf(stderr, "DL2 weights:\n");
        std::cerr << dl2.m_weight;
        fprintf(stderr, "DL2 bias:\n");
        std::cerr << dl2.m_bias;

        fprintf(stderr, "DL1 weights:\n");
        std::cerr << dl1.m_weight;
        fprintf(stderr, "DL1 bias:\n");
        std::cerr << dl1.m_bias;

        fprintf(stderr, "CL2 filter:\n");
        std::cerr << cl2.m_filter;
        fprintf(stderr, "CL2 bias:\n");
        std::cerr << cl2.m_bias;

        fprintf(stderr, "CL1 filter:\n");
        std::cerr << cl1.m_filter;
        fprintf(stderr, "CL1 bias:\n");
        std::cerr << cl1.m_bias;
        */

        // dl2.check_downstream_derivative(training_labels[0]);
        // dl2.check_weight_derivative(training_labels[0]);

        // dl1.check_downstream_derivative(training_labels[0]);
        // dl1.check_weight_derivative(training_labels[0]);

        // cl2.check_downstream_derivative(training_labels[0]);
        // cl2.check_weight_derivative(training_labels[0]);
        
        dl1.check_downstream_derivative(training_labels[0]);
        pl2.check_downstream_derivative(training_labels[0]);

        std::cerr << "PL2 input:" << std::endl;
        std::cerr << cl2.output;

        cl1.check_downstream_derivative(training_labels[0]);
        cl1.check_weight_derivative(training_labels[0]);

        pl1.check_downstream_derivative(training_labels[0]);
    }
}

void
testrun1() {

    /*
     * Read in training data.
     */

    static float training_images[60'000][28][28];
    read_mnist_images("mnist/train-images-idx3-ubyte", training_images);
    output_pgm("img0.pgm", training_images[0]);
    output_pgm("img59999.pgm", training_images[59999]);

    static unsigned char training_labels[60'000];
    read_mnist_labels("mnist/train-labels-idx1-ubyte", training_labels);
    assert(training_labels[0] == 5);
    assert(training_labels[59'999] == 8);

    {
        static InputLayer<Dims<1, 28, 28>> il;
        static ConvolutionalLayer<Dims<1, 28, 28>, 16> cl1("cl1", 1);
        static MaxPoolLayer<Dims<16, 28, 28>> pl1("pl1");
        static ConvolutionalLayer<Dims<16, 14, 14>, 16> cl2("cl2", 2);
        static MaxPoolLayer<Dims<16, 14, 14>> pl2("pl2");
        static FullyConnectedLayer<Dims<16, 7, 7>, 32> dl1("dl1", true, 0, 1);
        static FullyConnectedLayer<Dims<1, 1, 32>, 10> dl2("dl2", false, 0, 2);
        static SoftmaxLayer<10> sm;
        static CrossEntropyLayer<10> ce;

        il.next_layer = &cl1; cl1.previous_layer = &il;
        cl1.next_layer = &pl1; pl1.previous_layer = &cl1;
        pl1.next_layer = &cl2; cl2.previous_layer = &pl1;
        cl2.next_layer = &pl2; pl2.previous_layer = &cl2;
        pl2.next_layer = &dl1; dl1.previous_layer = &pl2;
        dl1.next_layer = &dl2; dl2.previous_layer = &dl1;
        dl2.next_layer = &sm; sm.previous_layer = &dl2;
        sm.next_layer = &ce; ce.previous_layer = &sm;

        il.train(training_images[0], training_labels[0], 1);

        fprintf(stderr, "DL2 weights:\n");
        std::cerr << dl2.m_weight;
        fprintf(stderr, "DL2 bias:\n");
        std::cerr << dl2.m_bias;

        fprintf(stderr, "DL1 weights:\n");
        std::cerr << dl1.m_weight;
        fprintf(stderr, "DL1 bias:\n");
        std::cerr << dl1.m_bias;

        fprintf(stderr, "CL2 filter:\n");
        std::cerr << cl2.m_filter;
        fprintf(stderr, "CL2 bias:\n");
        std::cerr << cl2.m_bias;

        fprintf(stderr, "CL1 filter:\n");
        std::cerr << cl1.m_filter;
        fprintf(stderr, "CL1 bias:\n");
        std::cerr << cl1.m_bias;

        /*
        dl2.check_downstream_derivative(training_labels[0]);
        dl2.check_weight_derivative(training_labels[0]);

        dl1.check_downstream_derivative(training_labels[0]);
        dl1.check_weight_derivative(training_labels[0]);

        cl2.check_downstream_derivative(training_labels[0]);
        cl2.check_weight_derivative(training_labels[0]);

        cl1.check_downstream_derivative(training_labels[0]);
        cl1.check_weight_derivative(training_labels[0]);
        */
    }
}

void
run1() {

    /*
     * Read in training data.
     */

    static float training_images[60'000][28][28];
    read_mnist_images("mnist/train-images-idx3-ubyte", training_images);
    output_pgm("img0.pgm", training_images[0]);
    output_pgm("img59999.pgm", training_images[59999]);

    static unsigned char training_labels[60'000];
    read_mnist_labels("mnist/train-labels-idx1-ubyte", training_labels);
    assert(training_labels[0] == 5);
    assert(training_labels[59'999] == 8);

    {
        static InputLayer<Dims<1, 28, 28>> il;
        static ConvolutionalLayer<Dims<1, 28, 28>, 32> cl1("cl1", 1);
        static MaxPoolLayer<Dims<32, 28, 28>> pl1("pl1");
        static ConvolutionalLayer<Dims<32, 14, 14>, 64> cl2("cl2", 2);
        static MaxPoolLayer<Dims<64, 14, 14>> pl2("pl2");
        static FullyConnectedLayer<Dims<64, 7, 7>, 1024> dl1("dl1", true, 0, 1);
        static FullyConnectedLayer<Dims<1, 1, 1024>, 10> dl2("dl2", false, 0, 2);
        static SoftmaxLayer<10> sm;
        static CrossEntropyLayer<10> ce;

        il.next_layer = &cl1; cl1.previous_layer = &il;
        cl1.next_layer = &pl1; pl1.previous_layer = &cl1;
        pl1.next_layer = &cl2; cl2.previous_layer = &pl1;
        cl2.next_layer = &pl2; pl2.previous_layer = &cl2;
        pl2.next_layer = &dl1; dl1.previous_layer = &pl2;
        dl1.next_layer = &dl2; dl2.previous_layer = &dl1;
        dl2.next_layer = &sm; sm.previous_layer = &dl2;
        sm.next_layer = &ce; ce.previous_layer = &sm;

        for (int r = 0; r < 100; r++) {

            if (r % 5 == 0) {

                int correct = 0;
                for (size_t i = 0; i < 100; i++) {
                    // fprintf(stderr, "Predict: %d for %lu\n", input.predict(training_images[i]), i);
                    if (il.predict(training_images[i]) == training_labels[i]) {
                        correct++;
                    }
                }
                fprintf(stderr, "Round %d: accuracy=%f\n", r, correct/100.0);

                for (size_t i = 0; i < 32; i++) {
                    char buf[100];
                    sprintf(buf, "r%02d-%02zu.pgm", r, i);
                    output_pgm(buf, cl1.m_filter[i]);
                }
                std::cerr << cl1.m_filter;
            }

            /*
            std::cerr << "Weights:" << std::endl;
            for (size_t n = 0; n < 10; n++) {
                std::cerr << "Neuron " << n << ":" << std::endl;
                print(std::cout, dl.weight[n]);
            }
            */

            for (size_t i = 0; i < 100; i++) {
                il.train(training_images[i], training_labels[i], 100);
            }
            il.update_weights(.01);
        }
    }
}

void
run2() {

    static float training_images[60'000][28][28];
    read_mnist_images("mnist/train-images-idx3-ubyte", training_images);
    output_pgm("img0.pgm", training_images[0]);
    output_pgm("img59999.pgm", training_images[59999]);

    static unsigned char training_labels[60'000];
    read_mnist_labels("mnist/train-labels-idx1-ubyte", training_labels);
    assert(training_labels[0] == 5);
    assert(training_labels[59'999] == 8);

    static float test_images[10'000][28][28];
    read_mnist_images("mnist/t10k-images-idx3-ubyte", test_images);
    static unsigned char test_labels[10'000];
    read_mnist_labels("mnist/t10k-labels-idx1-ubyte", test_labels);

    InputLayer<Dims<1, 28, 28>> il;
    FullyConnectedLayer<Dims<1, 28, 28>, 10> dl("dl", false, 0, 1);
    SoftmaxLayer<10> sm;
    CrossEntropyLayer<10> ce;

    il.next_layer = &dl; dl.previous_layer = &il;
    dl.next_layer = &sm; sm.previous_layer = &dl;
    sm.next_layer = &ce; ce.previous_layer = &sm;

    std::default_random_engine eng(9815);
    std::uniform_int_distribution<size_t> pick_test(0, 9'999);

    for (int e = 0; e < 100; e++) {

        // Create shuffled sequence of training images.
        std::vector<int> training(60'000);
        std::iota(training.begin(), training.end(), 0);
        assert(*--training.end() == 59'999);
        std::shuffle(training.begin(), training.end(), eng);

        for (int r = 0; r < 600; r++) {

            if (r%100 == 0) {

                // fprintf(stderr, "Begin predict...."); fflush(stderr);
                int correct = 0;
                for (size_t i = 0; i < 1000; i++) {
                    // fprintf(stderr, "Predict: %d for %lu\n", il.predict(training_images[i]), i);
                    size_t ind = pick_test(eng);
                    if (il.predict(test_images[ind]) == test_labels[ind]) {
                        correct++;
                    }
                }
                fprintf(stderr, "Epoch %d: Round %d: accuracy=%f\n", e, r, correct/1000.0);
            }

            /*
            std::cerr << "Weights:" << std::endl;
            for (size_t n = 0; n < 10; n++) {
                std::cerr << "Neuron " << n << ":" << std::endl;
                print(std::cout, dl.weight[n]);
            }
            */

            for (size_t i = 0; i < 100; i++) {
                il.train(training_images[training.at(100*r + i)], training_labels[training.at(100*r + i)], 100);
            }
            il.update_weights(.005);
        }
    }

    #if 0
    for (int r = 0; r < 100'000; r++) {

        if (r % 100 == 0) {

            int correct = 0;
            for (size_t i = 0; i < 100; i++) {
                // fprintf(stderr, "Predict: %d for %lu\n", input.predict(training_images[i]), i);
                if (input.predict(training_images[i]) == training_labels[i]) {
                    correct++;
                }
            }
            printf("Round %d: accuracy=%f\n", r, correct/100.0);
        }

        /*
        std::cerr << "Weights:" << std::endl;
        for (size_t n = 0; n < 10; n++) {
            std::cerr << "Neuron " << n << ":" << std::endl;
            print(std::cout, dl.weight[n]);
        }
        */

        for (size_t i = 0; i < 100; i++) {
            input.train(training_images[i], training_labels[i], 100);
        }
        input.update_weights(.001);
    }
    #endif
}

// vim: set tw=110:
