@testset "HBMatrix lower block-Toeplitz indexing" begin
        B1 = [1 2; 3 4]
        B2 = [5 6; 7 8]
        B3 = [9 10; 11 12]
        mBI = hcat(B1, B2, B3)
        HB = BEAVARs.HBMatrix(mBI, 4)

        expected = zeros(Int, 8, 8)
        expected[1:2, 1:2] = B1
        expected[3:4, 1:2] = B2
        expected[3:4, 3:4] = B1
        expected[5:6, 1:2] = B3
        expected[5:6, 3:4] = B2
        expected[5:6, 5:6] = B1
        expected[7:8, 3:4] = B3
        expected[7:8, 5:6] = B2
        expected[7:8, 7:8] = B1

        @test Matrix(HB) == expected
        @test HB[1:2, 1:2] == B1
        @test HB[3:4, 1:2] == B2
        @test HB[5:6, 1:2] == B3
        @test HB[1:2, 3:4] == zeros(Int, 2, 2)

        selector = sparse([1, 6, 7], [1, 2, 3], [1.0, 1.0, 1.0], 8, 3)
        product = zeros(Float64, 8, 3)
        @test mul!(product, HB, selector) ≈ Matrix(HB) * Matrix(selector)

        weighted_selector = sparse([1, 6, 7], [1, 2, 3], [2.0, -1.0, 0.5], 8, 3)
        initial = fill(0.25, 8, 3)
        product .= initial
        @test mul!(product, HB, weighted_selector, 1.5, -2.0) ≈
              1.5 .* Matrix(HB) * Matrix(weighted_selector) .- 2.0 .* initial
    end

    @testset "sparse adjoint times block-banded matrix" begin
        n, T = 2, 3
        A = sparse(
            [1, 3, 5, 2, 4, 6],
            [1, 1, 1, 2, 3, 4],
            [1.0, -2.0, 0.5, 3.0, -1.5, 2.0],
            T*n, 4,
        )
        diagonal = BlockBandedMatrix{Float64}(
            undef, fill(n, T), fill(n, T), (0, 0),
        )
        for t in 1:T
            diagonal[Block(t, t)] .= [1.0+t 0.25; -0.5 2.0-t]
        end

        product = fill(0.25, size(A, 2), size(A, 1))
        initial = copy(product)
        @test mul!(product, A', diagonal) ≈ Matrix(A') * Matrix(diagonal)
        product .= initial
        @test mul!(product, A', diagonal, 1.5, -2.0) ≈
              1.5 .* Matrix(A') * Matrix(diagonal) .- 2.0 .* initial

        banded = BlockBandedMatrix{Float64}(
            undef, fill(n, T), fill(n, T), (1, 1),
        )
        for t in 1:T, s in max(1, t-1):min(T, t+1)
            banded[Block(t, s)] .= [t+s 0.25; -0.5 t-s]
        end
        @test mul!(product, A', banded) ≈ Matrix(A') * Matrix(banded)
    end

    @testset "block array times sparse matrix" begin
        A = BlockedArray(
            [1.0 2.0 0.0 0.0 3.0 4.0;
             5.0 6.0 0.0 0.0 7.0 8.0;
             0.0 0.0 9.0 10.0 0.0 0.0;
             0.0 0.0 11.0 12.0 0.0 0.0],
            [2, 2], [2, 2, 2],
        )
        B = sparse(
            [1, 3, 5, 2, 4, 6],
            [1, 1, 1, 2, 3, 3],
            [1.0, -2.0, 0.5, 3.0, -1.5, 2.0],
            6, 3,
        )
        product = fill(0.25, size(A, 1), size(B, 2))
        initial = copy(product)
        @test mul!(product, A, B) ≈ Matrix(A) * Matrix(B)
        product .= initial
        @test mul!(product, A, B, 1.5, -2.0) ≈
              1.5 .* Matrix(A) * Matrix(B) .- 2.0 .* initial
    end