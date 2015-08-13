module MLBasePlotting

export plotperf

using MLBase
using Gadfly

# package code goes here
function plotperf(truth::AbstractVector, predictor::AbstractVector, curveType="pr"; name="", resolution=600)
    plotperf({"predictor" => (truth, predictor)}, curveType, name=name, resolution=resolution)
end
function plotperf(methods, curveType="pr"; name="", resolution=600)

    if curveType == "roc"
        xlabel = "False Positive Rate"
        ylabel = "True Positive Rate"
        xmap = x->false_positive_rate(x)
        ymap = x->true_positive_rate(x)
        if name == "" name = "ROC" end
    elseif curveType == "pr"
        xlabel = "Recall"
        ylabel = "Precision"
        xmap = x->recall(x)
        ymap = x->precision(x)
        if name == "" name = "Precision/Recall" end
    end
    labels = ASCIIString[]
    xdata = Float64[]
    ydata = Float64[]
    data = Dict()
    truth = methods[first(keys(methods))][1]
    methods["random"] = (truth, rand(length(truth)))
    for (key,(truth,predictor)) in methods
        rocData = MLBase.roc(int(truth), float(predictor), resolution)
        xvals = map(xmap, rocData)
        yvals = map(ymap, rocData)
        aucValue = @sprintf("%0.03f", PrescienceUtils.area_under_curve(reverse(xvals), reverse(yvals)))
        append!(labels, [repeat(["AUC = $aucValue, $key"], inner=[length(xvals)])])
        append!(xdata, xvals)
        append!(ydata, yvals)
    end

    plot(x=xdata, y=ydata, color=labels,
        Guide.title("$name"),
        Guide.XLabel(xlabel),
        Guide.YLabel(ylabel), Geom.line,
        Scale.discrete_color_manual(["grey", "blue", "red", "green", "purple", "pink", "orange"][1:length(methods)+1]...),
        Guide.colorkey("Methods")
    )
end

end # module
