struct Dual{T, N} <: Number
    value::T
    partials::Array{T, N}
end

value(x) = x
value(d::Dual) = d.value

partials(d::Dual) = d.partials

Dual(v::T, p::Array{T, N}) where {T<:Number, N} = Dual{T, N}(v, p)
Dual(v::T, p::Array{T2, N}) where {T<:Number, T2, N} = Dual{T, N}(v, Array{T}(p))

Base.:(==)(l::Dual, r::Dual) = value(l) == value(r) && partials(l) == partials(r)
Base.isapprox(l::Dual, r::Dual) = value(l) ≈ value(r) && partials(l) ≈ partials(r)

function onehot(T, dims, I)
    A = zeros(T, dims)
    A[I...] = one(T)

    A
end

dualize(x::T) where T <: Number = Dual(x, [one(T)])
function dualize(xs::Number...)
    nseeds = length(xs)
    (Dual(x, onehot(typeof(x), nseeds, i)) for (i, x) in enumerate(xs))
end

# Base.:+(l::Number, r::Dual)   = Dual(      l  + value(r), partials(r)                 )
# Base.:+(l::Dual,   r::Number) = Dual(value(l) +       r , partials(l)                 )
# Base.:+(l::Dual,   r::Dual)   = Dual(value(l) + value(r), partials(d1) .+ partials(d2))
#
# Base.:-(l::Number, r::Dual)   = Dual(      l  - value(r), partials(r)                 )
# Base.:-(l::Dual,   r::Number) = Dual(value(l) -       r , partials(l)                 )
# Base.:-(l::Dual,   r::Dual)   = Dual(value(l) - value(r), partials(d1) .- partials(d2))
#
# Base.:*(l::Number, r::Dual)   = Dual(      l  * value(r),          l  * partials(r))
# Base.:*(l::Dual,   r::Number) = Dual(value(l) *       r , partials(l) *          r )
# Base.:*(l::Dual,   r::Dual)   = Dual(value(l) * value(r), partials(l) * value(r) .+
#                                                             value(l) * partials(r))
#
# Base.:/(l::Number, r::Dual)   = Dual(value(l) / value(r), value(l) * partials(r))
# Base.:/(l::Dual,   r::Number) = Dual(value(l) / value(r), partials(l) / value(l))
# Base.:/(l::Dual,   r::Dual)   = Dual(value(l) / value(r), partials(l)*value(r) .+ value(l)*partials(r))
#
# Base.log(x::Dual) = Dual(log(value(x)), 1 ./ partials(x))

@testset begin
    @testset "basic arithmetic" begin
        d1, d2 = dualize(0.5, 2.0)
        @test d1 + d2 ≈ Dual(2.5, [1, 1])
        @test d1 - d2 ≈ Dual(-1.5, [1, -1])
        @test d1 * d2 ≈ Dual(1.0, [2.0, 0.5])
        @test d1 / d2 ≈ Dual(0.25, [0.5, -0.125])
    end

    @testset "trig functions" begin
        x = dualize(0.8)
        xv = value(x)
        @test sin(x) ≈ Dual(sin(xv), [cos(xv)])
    end

    @testset "function composition" begin
        # check for both real and complex-valued args
        # this doesn't require any complex-specific code in the AD, it just
        # falls out of the math when all functions are holomorphic
        for (x,y) in ((0.2, 1.5), (0.4+1.2im, -0.8+0.1im))
            x, y = dualize(0.2, 1.5)
            xv, yv = value(x), value(y)

            f(x, y) = x*(2x+sin(y))
            # manually define derivatives
            dfdx(x, y) = 4x+sin(y)
            dfdy(x, y) = x*cos(y)
            z = f(x, y)
            @test z ≈ Dual(f(xv, yv), [dfdx(xv, yv), dfdy(xv, yv)])
        end
    end
end

unary_ops = [:sin, :cos, :tan, :log]
for op in unary_ops
    @eval function Base.$op(d::Dual)
        x = value(d)
        diff, = diffrule($op)(x)

        # TODO: handle non-commutative partial multiplication
        Dual($op(x), diff .* partials(d))
    end
end

binary_ops = [:+, :-, :*, :/, :^]
for op in binary_ops
    @eval function Base.$op(l::Number, r::Dual)
        lx, rx = l, value(r)
        ldiff, rdiff = diffrule($op)(lx, rx)

        Dual($op(lx, rx), rdiff .* partials(r))
    end
    @eval function Base.$op(l::Dual, r::Number)
        lx, rx = value(l), r
        ldiff, rdiff = diffrule($op)(lx, rx)

        Dual($op(lx, rx), ldiff .* partials(l))
    end
    @eval function Base.$op(l::Dual, r::Dual)
        lx, rx = value(l), value(r)
        ldiff, rdiff = diffrule($op)(lx, rx)

        Dual($op(lx, rx), ldiff .* partials(l) + rdiff .* partials(r))
    end
end

# wrapper for a function that returns (dfdz, dfdz̄), necessary for
# differentiating non-holomorphic complex-valued functions
struct Wirtinger{F<:Function}
    f::F
end

@inline (w::Wirtinger)(args...) = w.f(args...)

# each diffrule returns a function that returns a tuple of partial derivatives

# unary functions
diffrule(::typeof(sin)) = (x)->(cos(x),)
diffrule(::typeof(cos)) = (x)->(-sin(x),)
diffrule(::typeof(tan)) = (x)->(sec(x)^2,)
diffrule(::typeof(log)) = (x)->(inv(x),)

# binary functions
diffrule(::typeof(+)) = (l, r)->(1, 1)
diffrule(::typeof(-)) = (l, r)->(1, -1)
diffrule(::typeof(*)) = (l, r)->(r, l)
diffrule(::typeof(/)) = (l, r)->(inv(r), -l/r^2)
diffrule(::typeof(^)) = (b, e)->(e*n^(e-1), b^e*log(b))

# non-holomorphic functions. each partial derivative is represented by a
# pair of partials with respect to z and conj(z)
diffrule(::typeof(abs2)) = Wirtinger((x)->((conj(x), x),))
diffrule(::typeof(conj)) = Wirtinger((x)->((0, conj(x),)))
