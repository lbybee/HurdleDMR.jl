reload("HurdleDMR")

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

include("testutils.jl")

using Distributions

ENV["OPENBLAS_NUM_THREADS"] = 1 # prevents thrashing by pmap + blas

# travis-ci limits to 4 or so
const nw = Sys.CPU_CORES-2
if nworkers() < nw
    if nworkers() > 1
        info("Removing existing parallel workers for tests...")
        rmprocs(workers())
    end
    info("Starting $nw parallel workers for tests...")
    addprocs(nw)
    info("$(nworkers()) parallel workers started")
end

using CSV, GLM, Lasso, DataFrames

import HurdleDMR; @everywhere using HurdleDMR

n = 100
p = 3
d = 100

srand(13)
m = 1+rand(Poisson(d/5),n)
covars = rand(n,p)
ηfn(vi) = exp.([0 + i/d*sum(vi) for i=1:d])
q = [ηfn(covars[i,:]) for i=1:n]
scale!.(q,ones(n)./sum.(q))
counts = convert(SparseMatrixCSC{Float64,Int},hcat(broadcast((qi,mi)->rand(Multinomial(mi, qi)),q,m)...)')

inposs = [1:p, 2:p]
inzeros = [1:p, 2:p]

# common args for all hdmr tests
testargs = Dict(:verbose=>false,:showwarnings=>false)

###########################################################
# hurdle with covarspos == covarszero
###########################################################
@testset "hurdle-dmr with covarspos == covarszero" begin

f = @model(h ~ v1 + v2 + vy, c ~ v1 + v2 + vy)
@test show(IOBuffer(),f) == nothing
projdir = findfirst(names(covarsdf),:vy)
dirpos = 3
dirzero = 3


local_cluster = true
parallel = true
inpos = inposs[1]
inzero = inzeros[1]
rtol = 1e-6

@time hdmrcoefs = hdmr(covars, counts; inpos=inpos, inzero=inzero, parallel=parallel, local_cluster=local_cluster, testargs...)
@time hdmroldcoefs = hdmrold(covars, counts; inpos=inpos, inzero=inzero, parallel=parallel, local_cluster=local_cluster, testargs...)
@test coef(hdmrcoefs)[1] ≈ coef(hdmroldcoefs)[1] rtol=rtol
@test coef(hdmrcoefs)[2] ≈ coef(hdmroldcoefs)[2] rtol=rtol

import Juno
Profile.init()
Profile.clear()
@profile hdmr(covars, counts; parallel=parallel, local_cluster=local_cluster, testargs...)
Juno.profiler()
Profile.clear()
@profile hdmrold(covars, counts; parallel=parallel, local_cluster=local_cluster, testargs...)
Juno.profiler()

@testset "hdmr vs hdmrold $deg" for deg=[:normal,:deg1,:deg2]
  zcounts = deepcopy(counts)
  if deg == :deg1
    # column j is never zero, so hj=1 for all observations
    zcounts[:,2] = zeros(n)
    zcounts[:,3] = ones(n)
  elseif deg == :deg2
    srand(13)
    for I = eachindex(zcounts)
        if iszero(zcounts[I])
            zcounts[I] = rand(1:10)
        end
    end
  end
  # # make sure we are not adding all zero obseravtions
  # m = sum(zcounts,2)
  # @test sum(m .== 0) == 0

  @testset "local_cluster=$local_cluster" for local_cluster=[true,false][1:1]
    @testset "parallel=$parallel" for parallel=[true,false][1:2]
      @testset "inpos=$inpos,inzero=$inzero" for inpos=inposs, inzero=inzeros
        hdmrcoefs, t, bytes, gctime, memallocs = @timed hdmr(covars, zcounts; inpos=inpos, inzero=inzero, parallel=parallel, local_cluster=local_cluster, testargs...)
        hdmroldcoefs, told, bytesold, gctimeold, memallocsold = @timed hdmrold(covars, zcounts; inpos=inpos, inzero=inzero, parallel=parallel, local_cluster=local_cluster, testargs...)
        speedup = 100*(told/t - 1)
        memeffic = 100*(bytesold/bytes - 1)
        info("$speedup% speedup, $memeffic% memory efficiency improvment")
        @test coef(hdmrcoefs)[1] ≈ coef(hdmroldcoefs)[1] rtol=rtol
        @test coef(hdmrcoefs)[2] ≈ coef(hdmroldcoefs)[2] rtol=rtol
      end
    end
  end
end
