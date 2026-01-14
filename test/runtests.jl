using Test
using GalaxyCNN

@testset "GalaxyCNN tests" begin
    include("metric_test.jl")
    include("translation_test.jl")
end