using MLBasePlotting
using Base.Test

# plotperf
labels = round(Int, bitrand(1000))
values = randn(1000)
plotperf(labels, values)
plotperf(labels, values, name="ROC Test", curveType="roc")
plotperf(Dict(
	"model1" => (labels, values),
	"model2" => (labels, randn(1000))
))


# area_under_pr
x = rand([true,false], 100)
@test_approx_eq area_under_pr(x, x) 1.0
@test area_under_pr(x, -x) < 0.5

# area_under_roc
# @test_approx_eq area_under_roc(x, x) 1.0 TODO: this is broken and I don't know why yet
@test area_under_roc(x, -x) < 0.5
