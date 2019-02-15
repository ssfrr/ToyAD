using PedagogicalAutoDiff
using PedagogicalAutoDiff: CtoR, NonHolomorphic, AntiHolomorphic
using PedagogicalAutoDiff: wirtprimal, wirtconj
using Test

# define a nonholomorphic function and its derivatives.
nonholo(z) = 3z^2 * 5conj(z)^3
dnonholodz(z) = 6z * 5conj(z)^3
dnonholodz̄(z) = 3z^2 * 15conj(z)^2
# define an explicit diff rule so we're not testing any function composition
@partials nonholo(z) = (NonHolomorphic(dnonholodz(z), dnonholodz̄(z)),)

# TODO: we should export a macro to do these registrations
function nonholo(d::Dual)
    x = value(d)
    diff, = PedagogicalAutoDiff.diffrule(nonholo)(x)

    # use Ref to treat diff as a scalar in broadcast
    Dual(nonholo(x), PedagogicalAutoDiff.forwardprop.(Ref(diff), partials(d)))
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
    z = 10.0+1.0im
    dualz = dualseed(z)
    dualf = abs2(dualz)
    dfdz = partials(dualf)[1]
    @test value(dualf) ≈ abs2(z)
    @test dfdz isa CtoR
    @test wirtprimal(dfdz) ≈ conj(z)
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
    @test wirtprimal(df) ≈ dnonholodz(z)
    @test wirtconj(df) ≈ conj(dnonholodz̄(z))
end
end
