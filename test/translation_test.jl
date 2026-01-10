using Test
include("../src/augmentation/translation.jl")

@testset "translate!" begin

    @testset "zero translation" begin
        # Simple 3×4 single-channel image
        img = reshape(collect(1:9), 3, 3, 1)
        out = similar(img)

        translate!(out, img, 0, 0)

        @test out == img
    end

    @testset "positive shift dx=1, dy=1 with reflection" begin
        # 3×3 single-channel image with known values
        img = reshape(collect(1:9), 3, 3, 1)
        out = similar(img)

        translate!(out, img, 1, 1)

        expected = reshape([
            5 2 5;
            4 1 4;
            5 2 5
        ], 3, 3, 1)

        @test out == expected
    end

    @testset "negative shift dx=-1, dy=-1 with reflection" begin
        img = reshape(collect(1:9), 3, 3, 1)
        out = similar(img)

        translate!(out, img, -1, -1)

        expected = reshape([
            5 8 5;
            6 9 6;
            5 8 5
        ], 3, 3, 1)

        @test out == expected
    end

    @testset "multiple channels" begin
        # Two-channel 2×2 image
        img = Array{Int}(undef, 2, 2, 2)
        img[:, :, 1] = [1 2; 3 4]
        img[:, :, 2] = [10 20; 30 40]

        out = similar(img)

        translate!(out, img, 0, 0)

        @test out == img
    end

    @testset "invalid output size" begin
        img = zeros(3, 3, 1)
        out = zeros(4, 3, 1)

        @test_throws AssertionError translate!(out, img, 0, 0)
    end

    @testset "translation exceeds allowed range" begin
        img = zeros(4, 4, 1)
        out = similar(img)

        # Translation larger than half the image size must fail
        @test_throws AssertionError translate!(out, img, 3, 0)
        @test_throws AssertionError translate!(out, img, 0, 3)
    end

end
