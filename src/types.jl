abstract type Wirtinger end

function wirtshow(io, w::Wirtinger)
    print(io, "(dz: $(wirtprimal(w)), dz̄: $(wirtconj(w)))")
end

# In the most general non-holomorphic case, we need to track both derivatives
# (dfdz, dfdz̄), necessary for differentiating non-holomorphic complex-valued
# functions
struct NonHolomorphic{PT, CT} <: Wirtinger
    primal::PT
    conjugate::CT
end

function Base.show(io::IO, w::NonHolomorphic)
    print(io, "NonHolomorphic")
    wirtshow(io, w)
end

function Base.convert(::Type{<:Complex}, w::Wirtinger)
    wirtconj(w) ≈ 0.0 || throw(DomainError(w, "The conjugate component must be zero"))
    wirtprimal(w)
end

# for the derivatives of real-valued functions the conjugate partial derivatives
# are not independent of the primal, they are always the conjugate
struct CtoR{T} <: Wirtinger
    primal::T
end

function Base.show(io::IO, w::CtoR)
    print(io, "CtoR")
    wirtshow(io, w)
end

struct AntiHolomorphic{T} <: Wirtinger
    conjugate::T
end

function Base.show(io::IO, w::AntiHolomorphic)
    print(io, "AntiHolomorphic")
    wirtshow(io, w)
end

Base.:+(l::NonHolomorphic, r::NonHolomorphic) =
    NonHolomorphic(l.primal + r.primal, l.conjugate + r.conjugate)

Base.:+(l::CtoR, r::CtoR) =
    CtoR(l.primal + r.primal)

Base.:+(l::AntiHolomorphic, r::AntiHolomorphic) =
    AntiHolomorphic(l.conjugate + r.conjugate)

# fallback for when the types don't match.
Base.:+(l::Wirtinger, r::Wirtinger) =
    convert(NonHolomorphic, l) + convert(NonHolomorphic, r)

Base.:+(l::Wirtinger, r::Number) =
    convert(NonHolomorphic, l) + convert(NonHolomorphic, r)

Base.:+(l::Number, r::Wirtinger) =
    convert(NonHolomorphic, l) + convert(NonHolomorphic, r)

Base.convert(::Type{NonHolomorphic}, x::Union{Wirtinger, Number}) =
    NonHolomorphic(wirtprimal(x), wirtconj(x))

# we support promotion mostly so that addition works when we're summing up
# the partial differentials to get the total. It might be safer to just define
# all the addition methods we need.
# Base.promote_rule(::Type{<:NonHolomorphic}, ::Type{<:AntiHolomorphic}) = NonHolomorphic
# Base.promote_rule(::Type{<:NonHolomorphic}, ::Type{<:CtoR}) = NonHolomorphic
# Base.promote_rule(::Type{<:NonHolomorphic}, ::Type{<:Number}) = NonHolomorphic
# Base.promote_rule(::Type{<:AntiHolomorphic}, ::Type{<:CtoR}) = NonHolomorphic
# Base.promote_rule(::Type{<:AntiHolomorphic}, ::Type{<:Number}) = NonHolomorphic
# Base.promote_rule(::Type{<:CtoR}, ::Type{<:Number}) = NonHolomorphic

# This comes in handy in the propagation rules, because we can express the
# general case in terms of the partials and conjpartials, and eliminate
# unnecessary computation in a type-stable way.
struct Zero end
@inline Base.:+(::Zero, x) = x
@inline Base.:+(x, ::Zero) = x
@inline Base.:+(::Zero, ::Zero) = Zero()
@inline Base.:*(::Zero, x) = Zero()
@inline Base.:*(x, ::Zero) = Zero()
@inline Base.:*(::Zero, ::Zero) = Zero()
@inline Base.conj(::Zero) = Zero()
@inline Base.:(==)(::Zero, x) = 0 == x
@inline Base.:(==)(x, ::Zero) = x == 0
@inline Base.isapprox(::Zero, x) = 0 ≈ x
@inline Base.isapprox(x, ::Zero) = x ≈ 0
Base.show(io::IO, ::Zero) = print(io, "Zero()")

# Wirtinger unwrapping functions
@inline wirtprimal(x::NonHolomorphic) = x.primal
@inline wirtconj(x::NonHolomorphic) = x.conjugate
@inline wirtprimal(x::CtoR) = x.primal
@inline wirtconj(x::CtoR) = conj(x.primal)
@inline wirtprimal(::AntiHolomorphic) = Zero()
@inline wirtconj(x::AntiHolomorphic) = x.conjugate
@inline wirtprimal(x) = x
@inline wirtconj(::Any) = Zero()
