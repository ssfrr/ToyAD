using ToyAD
using ToyAD: CtoR, NonHolomorphic, AntiHolomorphic
using ToyAD: wirtprimal, wirtconj
using Test

# define some known functions that we can use to test function composition. We
# define partials for these so that they themselves aren't made of composed
# functions
@noinline holo(z) = z^2
dholo_dz(z) = 2z
dholo_dz̄(z) = 0
@partials holo(z) = (dholo_dz(z),)
@add_forward_unary holo

@noinline nonholo(z) = 3z^2 * 5conj(z)^3
dnonholo_dz(z) = 6z * 5conj(z)^3
dnonholo_dz̄(z) = 3z^2 * 15conj(z)^2
@partials nonholo(z) = (NonHolomorphic(dnonholo_dz(z), dnonholo_dz̄(z)),)
@add_forward_unary nonholo

@noinline antiholo(z) = conj(z^2)
dantiholo_dz(z) = 0
dantiholo_dz̄(z) = 2*conj(z)
@partials antiholo(z) = (AntiHolomorphic(dantiholo_dz̄(z)),)
@add_forward_unary antiholo

@noinline ctor(z) = conj(z)*z
dctor_dz(z) = conj(z)
dctor_dz̄(z) = z
@partials ctor(z) = (CtoR(dctor_dz(z)),)
@add_forward_unary ctor

# test a dual function
function testforward(fn, expectedT, dfdz=nothing, dfdz̄=nothing)
    z = 10.0+1.0im
    dualz = dualseed(z)
    dualf = fn(dualz)
    df = partials(dualf)[1]
    @test value(dualf) ≈ fn(z)
    @test df isa expectedT
    @test wirtprimal(df) ≈ dfdz(z)
    dfdz̄ !== nothing && @test wirtconj(df) ≈ dfdz̄(z)
end

@testset begin
@testset "basic arithmetic" begin
    d1, d2 = dualseed(0.5, 2.0)
    @test d1 + d2 ≈ Dual(2.5, (1, 1))
    @test d1 - d2 ≈ Dual(-1.5, (1, -1))
    @test d1 * d2 ≈ Dual(1.0, (2.0, 0.5))
    @test d1 / d2 ≈ Dual(0.25, (0.5, -0.125))
end

@testset "trig functions" begin
    x = dualseed(0.8)
    xv = value(x)
    @test sin(x) ≈ Dual(sin(xv), (cos(xv),))
end

@testset "function composition" begin
    # check for both real and complex-valued args
    # this doesn't require any complex-specific code in the AD, it just
    # falls out of the math when all functions are holomorphic
    for (xv,yv) in ((0.2, 1.5), (0.4+1.2im, -0.8+0.1im))
        x, y = dualseed(xv, yv)

        f(x, y) = x*(2x+sin(y))
        # manually define derivatives
        dfdx(x, y) = 4x+sin(y)
        dfdy(x, y) = x*cos(y)
        z = f(x, y)
        @test z ≈ Dual(f(xv, yv), (dfdx(xv, yv), dfdy(xv, yv)))
    end
end

@testset "CtoR" begin
    testforward(abs2, CtoR, conj)
end

@testset "AntiHolomorphic" begin
    z = 10.0+1.0im
    dualz = dualseed(z)
    dualf = conj(dualz)
    dfdz = partials(dualf)[1]
    @test value(dualf) ≈ conj(z)
    @test dfdz isa AntiHolomorphic
    @test wirtprimal(dfdz) ≈ 0
    @test wirtconj(dfdz) ≈ 1
end

@testset "Composed NonHolomorphic" begin
    z = 10.0+1.0im
    dualz = dualseed(z)

    # this time use a new function that's identical to nonholo, so we should
    # get the same result but this time force the system to compose basic
    # functions
    f(z) = 3z^2 * 5conj(z)^3
    dualf = f(dualz)
    df = partials(dualf)[1]
    @test value(dualf) ≈ nonholo(z)
    @test df isa NonHolomorphic
    @test wirtprimal(df) ≈ dnonholo_dz(z)
    @test wirtconj(df) ≈ conj(dnonholo_dz̄(z))
end

# test all compositions of function types
# @testset "Composition Matrix" begin
#     testforward(z->nonholo(nonholo(z)), NonHolomorphic, , )
#     testforward(z->nonholo(holo(z)), NonHolomorphic, , )
#     testforward(z->nonholo(antiholo(z)), NonHolomorphic, , )
#     testforward(z->nonholo(ctor(z)), NonHolomorphic, , )
#
#     testforward(z->holo(nonholo(z)), NonHolomorphic, , )
#     testforward(z->holo(holo(z)), Complex, 4z^3, 0)
#     testforward(z->holo(antiholo(z)), AntiHolomorphic, , )
#     testforward(z->holo(ctor(z)), NonHolomorphic, , )
#
#     testforward(z->antiholo(nonholo(z)), NonHolomorphic, , )
#     testforward(z->antiholo(holo(z)), AntiHolomorphic, , )
#     testforward(z->antiholo(antiholo(z)), Complex, , )
#     testforward(z->antiholo(ctor(z)), NonHolomorphic, , )
#
#     testforward(z->ctor(nonholo(z)), CtoR, , )
#     testforward(z->ctor(holo(z)), CtoR, , )
#     testforward(z->ctor(antiholo(z)), CtoR, , )
#     testforward(z->ctor(ctor(z)), CtoR, , )
# end

end
