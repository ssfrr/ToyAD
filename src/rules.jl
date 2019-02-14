# each diffrule returns a function that returns a tuple of partial derivatives
# a macro could make these rules easier to write

using MacroTools: splitdef

function diffrule end

"""
Convenience macro to specify partial differentiation rules. The given expression
looks like a function definition of the function you're defining a derivative
for, and should return a tuple of partial derivatives (one for each argument).

For example, to define the derivative of the `sin` and `*` functions:

    @partials sin(x)=(cos(x),)
    @partials *(l, r)=(r, l)

This is translated into

    diffrule(::typeof(sin)) = (x)->(cos(x), )
    diffrule(::typeof(*)) = (l, r)->(r, l)

You can test your diff rule:

    julia> diffrule(sin)(0)
    (1.0,)
    julia> diffrule(*)(4, 5)
    (5, 4)

For nonholomorphic functions involving Complex numbers, you'll need to wrap
the returned partial derivatives in types that encode that fact. For example,
given a function `foo(z) = conj(z)^2 * 2z^3`, you can instead reframe it as a
function `foo2(z, zc) = zc^2*2z^3`, and return the partial derivatives with
respect to `z` and `zc`:

    @partials foo(z) = (NonHolomorphic(conj(z)^2 * 6z^2, 2conj(z) * 2z^3), )

We call these the "primal" and "conjugate" derivatives.

Functions that take `Complex` numbers and return `Real` ones are by definition
not holomorphic, but they have the special structure that the conjugate
derivative (the derivative of `conj(z)`) is always conjugate to the primal
(the derivative of `z`), i.e. `d conj(z) == conj(dz)`. Because of this property
we only need to store the primal. For example, you can think of the `abs2`
function as being defined as `abs2(z) = conj(z)*z`. We can supply the derivative
as:

    @partials abs2(z) = (CtoR(conj(z)), )

The `CtoR` wrapper is similar to the `NonHolomorphic` one, but allows us to take
advantage of the conjugate relationship.
"""
macro partials(ex)
    def = splitdef(ex)

    fname = def[:name]
    args = esc.(def[:args])
    body = esc(def[:body])
    quote
        PedagogicalAutoDiff.diffrule(::typeof($(esc(fname)))) = ($(args...),) -> $body
    end
end

abstract type Wirtinger end

# In the most general non-holomorphic case, we need to track both derivatives
# (dfdz, dfdz̄), necessary for differentiating non-holomorphic complex-valued
# functions
struct NonHolomorphic{T} <: Wirtinger
    primal::T
    conjugate::T
end

function Base.:+(l::NonHolomorphic, r::NonHolomorphic)
    NonHolomorphic(partials(l) .+ partials(r), conjpartials(l) .+ conjpartials(r))
end

# for the derivatives of real-valued functions the conjugate partial derivatives
# are not independent of the primal, they are always the conjugate
struct CtoR{T} <: Wirtinger
    primal::T
end

function Base.:+(l::CtoR, r::CtoR)
    CtoR(partials(l) .+ partials(r))
end

struct AntiHolomorphic{T} <: Wirtinger
    conjugate::T
end

function Base.:+(l::AntiHolomorphic, r::AntiHolomorphic)
    AntiHolomorphic(conjpartials(l) .+ conjpartials(r))
end

# Base.convert(::Type{<:NonHolomorphic}, p::Partials) = NonHolomorphic(partials(p), conjpartials(p))
# Base.convert(::Type{<:NonHolomorphic}, x::Tuple) = NonHolomorphic(x, 0)



# Base.promote_type(::Type{<:NonHolomorphic}, ::Type{<:Partials}) = NonHolomorphic
# Base.promote_type(::Type{<:NonHolomorphic}, ::Type{<:Tuple}) = NonHolomorphic

# This comes in handy in the propagation rules, because we can express the
# general case in terms of the partials and conjpartials, and eliminate
# unnecessary computation in a type-stable way.
struct Zero end
@inline Base.:+(::Zero, x) = x
@inline Base.:+(x, ::Zero) = x
@inline Base.:*(::Zero, x) = Zero()
@inline Base.:*(x, ::Zero) = Zero()

# Wirtinger unwrapping functions
@inline wirtprimal(x) = x
@inline wirtconj(::Any) = Zero()
@inline wirtprimal(x::NonHolomorphic) = x.primal
@inline wirtconj(x::NonHolomorphic) = x.conjugate
@inline wirtprimal(x::CtoR) = x.primal
@inline wirtprimal(::AntiHolomorphic) = Zero()
@inline wirtconj(x::AntiHolomorphic) = x.conjugate

# this is not well-defined. If the wirtinger is being used as a perturbation
# then wirtconj(z) = z.primal, but if it's a partial derivative then
# wirtconj(z) = conj(z.primal).
@inline wirtconj(x::CtoR) = throw(ArgumentError("wirtconj not well-defined on CtoR types"))

# unary holomorphic functions
@partials sin(x) = (cos(x),  )
@partials cos(x) = (-sin(x), )
@partials tan(x) = (sec(x)^2,)
@partials log(x) = (inv(x),  )
@partials -(x) =   (-1,      )

# the FFT is just an efficient matmul by a constant matrix with special
# structure, so d(fft(x)) = fft(dx)
# @partials fft(x) = (fft, )
# @partials ifft(x) = (ifft, )
# @partials rfft(x) = (rfft, )
# @partials irfft(x) = (irfft, )

# binary holomorphic functions
@partials +(l, r) = (1,         1)
@partials -(l, r) = (1,        -1)
@partials *(l, r) = (r,         l)
@partials /(l, r) = (inv(r),   -l/r^2)
@partials ^(b, e) = (e*n^(e-1), b^e*log(b))

# for antiholomorphic functions we only store the conjugate partial derivative,
# because the primal is always zero. The composition of a holomorphic function
# and an antiholomorphic one is antiholomorphic
@partials conj(z) = (AntiHolomorphic(conj(z)),)

# C->R functions - we can take avantage of some special structure to save on
# computation. The conjugate partial is always conjugate to the primal, so we
# only need to specify the primal
@partials abs2(z) = (CtoR(conj(z)),)
@partials real(z) = (CtoR(0.5)    ,)
@partials imag(z) = (CtoR(1/2im)  ,)
