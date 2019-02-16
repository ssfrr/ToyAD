using ToyAD
using ToyAD: CtoR, NonHolomorphic, AntiHolomorphic
using ToyAD: wirtprimal, wirtconj
using Test

# define some known functions that we can use to test function composition. We
# define partials for these so that they themselves aren't made of composed
# functions
@noinline holo(z) = (2+im) * sin(z)
dholo_dz(z) = (2+im) * cos(z)
dholo_dz̄(z) = 0
@partials holo(z) = (dholo_dz(z),)
@add_forward_unary holo

@noinline nonholo(z) = (2+im) * z * z'^2
dnonholo_dz(z) = (2+im) * z'^2
dnonholo_dz̄(z) = (2+im) * z * 2z'
@partials nonholo(z) = (NonHolomorphic(dnonholo_dz(z), dnonholo_dz̄(z)),)
@add_forward_unary nonholo

@noinline antiholo(z) = (2+im) * conj(z^2)
dantiholo_dz(z) = 0
dantiholo_dz̄(z) = (2+im) * 2z'
@partials antiholo(z) = (AntiHolomorphic(dantiholo_dz̄(z)),)
@add_forward_unary antiholo

@noinline ctor(z) = z'*z
dctor_dz(z) = z'
dctor_dz̄(z) = z
@partials ctor(z) = (CtoR(dctor_dz(z)),)
@add_forward_unary ctor

# test a dual function
function testforward(fn, expectedT, dfdz=nothing, dfdz̄=nothing)
    z = 3.0+1.0im
    dualz = dualseed(z)
    dualf = fn(dualz)
    df = partials(dualf)[1]
    @test value(dualf) ≈ fn(z)
    @test df isa expectedT
    @test wirtprimal(df) ≈ dfdz(z)
    dfdz̄ !== nothing && @test wirtconj(df) ≈ dfdz̄(z)
end

@testset "ToyAD Tests" begin
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
    z = 3.0+1.0im
    dualz = dualseed(z)
    dualf = conj(dualz)
    dfdz = partials(dualf)[1]
    @test value(dualf) ≈ z'
    @test dfdz isa AntiHolomorphic
    @test wirtprimal(dfdz) ≈ 0
    @test wirtconj(dfdz) ≈ 1
end

@testset "Composed NonHolomorphic" begin
    z = 3.0+1.0im
    dualz = dualseed(z)

    # this time use a new function that's identical to nonholo, so we should
    # get the same result but this time force the system to compose basic
    # functions
    f(z) = (2+im) * z * conj(z)^2
    dualf = f(dualz)
    df = partials(dualf)[1]
    @test value(dualf) ≈ nonholo(z)
    @test df isa NonHolomorphic
    @test wirtprimal(df) ≈ dnonholo_dz(z)
    @test wirtconj(df) ≈ dnonholo_dz̄(z)
end

# test all compositions of function types
@testset "Composition Matrix" begin
    C = (2+im)
    testforward(z->nonholo(nonholo(z)), NonHolomorphic,
        z->C^2*C'^2*5z^4*z'^4, z->C^2*C'^2*4z^5*z'^3)
    testforward(z->nonholo(holo(z)), NonHolomorphic,
        z->C^2*C'^2*conj(sin(z))^2*cos(z),
        z->C^2*C'^2*2sin(z)*conj(sin(z))*conj(cos(z)))
    testforward(z->nonholo(antiholo(z)), NonHolomorphic,
        z->C^2*C'^2*4*z^3*z'^2,
        z->C^2*C'^2*2*z^4*z')
    testforward(z->nonholo(ctor(z)), NonHolomorphic,
        z->C*3*z^2*z'^3,
        z->C*3*z^3*z'^2)

    testforward(z->holo(nonholo(z)), NonHolomorphic,
        z->C^2*cos(C*z*z'^2)*z'^2,
        z->C^2*cos(C*z*z'^2)*2z'*z)
    testforward(z->holo(holo(z)), Complex,
        z->C^2*cos((C*sin(z)))cos(z),
        z->0)
    testforward(z->holo(antiholo(z)), AntiHolomorphic,
        z->0,
        z->C^2*cos(C*z'^2)*2z')
    testforward(z->holo(ctor(z)), NonHolomorphic,
        z->C*cos(z'*z)*z',
        z->C*cos(z'*z)*z)

    testforward(z->antiholo(nonholo(z)), NonHolomorphic,
        z->C*C'^2*4z'^2*z^3,
        z->C*C'^2*2z'*z^4)
    testforward(z->antiholo(holo(z)), AntiHolomorphic,
        z->0,
        z->C*C'^2*sin(2*z'))
    testforward(z->antiholo(antiholo(z)), Complex,
        z->C*C'^2*4z^3,
        z->0)
    testforward(z->antiholo(ctor(z)), NonHolomorphic,
        z->C*2z*z'^2,
        z->C*2z^2*z')

    testforward(z->ctor(nonholo(z)), CtoR,
        z->C'*C*3z'^3*z^2,
        z->C'*C*3z'^2*z^3)
    testforward(z->ctor(holo(z)), CtoR,
        z->C*C'*sin(z')cos(z),
        z->C*C'*cos(z')sin(z))
    testforward(z->ctor(antiholo(z)), CtoR,
        z->C'*C*2*z*z'^2,
        z->C'*C*2*z^2*z')
    testforward(z->ctor(ctor(z)), CtoR,
        z->2z'^2*z,
        z->2z'*z^2)
end

end
