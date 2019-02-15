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

In the future this will need to be modified to take the argument types count and
types into account.
"""
macro partials(ex)
    def = splitdef(ex)

    fname = def[:name]
    args = esc.(def[:args])
    body = esc(def[:body])
    quote
        ToyAD.diffrule(::typeof($(esc(fname)))) = ($(args...),) -> $body
    end
end

# unary holomorphic functions
@partials sin(x) = (cos(x),  )
@partials cos(x) = (-sin(x), )
@partials tan(x) = (sec(x)^2,)
@partials log(x) = (inv(x),  )
# this conflicts with subtraction - we need to include the arg types in the
# diffrule lookup
# @partials -(x) =   (-1,      )

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
@partials ^(b, e) = (e*b^(e-1), b^e*log(b))

# for antiholomorphic functions we only store the conjugate partial derivative,
# because the primal is always zero. The composition of a holomorphic function
# and an antiholomorphic one is antiholomorphic
@partials conj(z) = (AntiHolomorphic(1),)

# C->R functions - we can take avantage of some special structure to save on
# computation. The conjugate partial is always conjugate to the primal, so we
# only need to specify the primal
@partials abs2(z) = (CtoR(conj(z)),)
@partials real(z) = (CtoR(0.5)    ,)
@partials imag(z) = (CtoR(1/2im)  ,)
