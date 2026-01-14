using Test
using GalaxyCNN


@testset "l2_normalize" begin

    @testset "unit norm dims=1 test" begin
        x = randn(Float32, 4, 5)
        y = l2_normalize(x; dims=1)

        norms = sqrt.(sum(y.^2, dims=1))
        @test all(isapprox.(norms, 1f0; atol=1e-5))
    end

    @testset "unit norm dims=2 test" begin
        x = randn(Float32, 4, 5)
        y = l2_normalize(x; dims=2)

        norms = sqrt.(sum(y.^2, dims=2))
        @test all(isapprox.(norms, 1f0; atol=1e-5))
    end

    @testset "zero vectors test" begin
        x = zeros(Float32, 3, 4)
        y = l2_normalize(x)

        @test all(isfinite, y)
        @test all(y .== 0f0)
    end

    @testset "small vectors test" begin
        x = fill(1e-12, 3, 4)
        y = l2_normalize(x)

        norms = sqrt.(sum(y.^2, dims=1))
        @test all(isfinite, y)
        @test all(abs.(y) .< 1e6)
    end

end


@testset "distances" begin

    function diag(A::AbstractMatrix)
        return [A[i,i] for i in 1:min(size(A)...)]
    end

    @testset "distance matrix properties test" begin
        emb = randn(Float32, 8, 6)
        D = distances(emb)

        # Square matrix
        @test size(D) == (6, 6)

        # Symmetry
        @test D ≈ D'

        # Zero diagonal
        @test all(isapprox.(diag(D), 0.0; atol=1e-6))

        # Non-negativity
        @test all(D .>= 0)
    end

    @testset "identical embeddings test" begin
        emb = ones(Float32, 4, 3)
        D = distances(emb)

        @test all(isapprox.(D, 0.0; atol=1e-6))
    end

end


@testset "contrastive_loss" begin

    @testset "zero loss test" begin
        emb = Float32[
            1  1  -1 -1;
            0  0   0  0
        ]

        labels = [1, 1, 2, 2]
        D = distances(emb)

        loss = metric_loss_fn(ContrastiveLoss(0.7f0, 1.6f0), D, labels)
        @test isapprox(loss, 0f0; atol=1e-5)
    end

    @testset "positive loss test" begin
        emb = randn(Float32, 4, 6)
        labels = [1, 1, 1, 2, 2, 2]
        D = distances(emb)

        loss = metric_loss_fn(ContrastiveLoss(0.7f0, 1.6f0), D, labels)
        @test loss ≥ 0f0
    end

end


@testset "triplet_loss" begin

    @testset "zero loss test" begin
        emb = Float32[
            1   1   -1  -1;
            0   0    0   0
        ]

        labels = [1, 1, 2, 2]
        D = distances(emb)

        loss = metric_loss_fn(TripletLoss(0.4f0, 0.8f0), D, labels)
        @test isapprox(loss, 0f0; atol=1e-5)
    end

    @testset "non-negative loss test" begin
        emb = randn(Float32, 8, 5)
        labels = [1, 1, 2, 2, 3]
        D = distances(emb)

        loss = metric_loss_fn(TripletLoss(0.4f0, 0.8f0), D, labels)
        @test loss ≥ 0f0
    end

end

