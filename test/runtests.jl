using PedagogicalAutoDiff
using Test

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

    @testset "nonholomorphic functions" begin
    end
end

x, y, z = dualseed(1, 2, 3)
x
y
z

2*x
x+y
x*3
