# common args for all multicounts tests
testargs = Dict(:γ=>γ, :λminratio=>0.01, :verbose=>false,:showwarnings=>true)

counts1 = counts[:,:]
counts2 = counts1[:,:]
counts2[2:end,:] += counts1[1:end-1,:]
counts3 = counts2[:,:]
counts3[3:end,:] += counts1[1:end-2,:]
multicounts = [counts1, counts2, counts3]

@testset "mcdmr" begin

dmrcoefs = dmr(covars, multicounts[1]; testargs...)

Z, multicoefs = mcdmr(covars, multicounts, projdir; testargs...)

coefs = coef(dmrcoefs)
@test size(coefs) == (p+1, d)
@test coef(multicoefs[1]) == coefs
@test size(coef(multicoefs[2]),2) == size(coefs,2)
@test size(coef(multicoefs[2]),1) == size(coefs,1) + 2
@test size(coef(multicoefs[3]),2) == size(coefs,2)
@test size(coef(multicoefs[3]),1) == size(coefs,1) + 4

end
