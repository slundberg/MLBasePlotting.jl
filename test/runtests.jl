using MLBasePlotting
using Base.Test

# plotperf
labels = int(randbool(1000))
values = randn(1000)
plotperf(labels, values)
plotperf(labels, values, name="ROC Test", curveType="roc")
plotperf(Dict(
	"model1" => (labels, values),
	"model2" => (labels, randn(1000))
))
