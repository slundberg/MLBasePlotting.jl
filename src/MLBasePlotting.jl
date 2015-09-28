module MLBasePlotting

export plotperf

using MLBase
using Gadfly

function plotperf(sortedTruth::AbstractVector; curveType="pr", name="", resolution=600)
    methods = Dict()
    methods["predictor"] = (sortedTruth, -collect(1:length(sortedTruth)))
    plotperf(methods, curveType=curveType, name=name, resolution=resolution)
end
function plotperf(truth::AbstractVector, predictor::AbstractVector; curveType="pr", name="", resolution=600)
    methods = Dict()
    methods["predictor"] = (truth, predictor)
    plotperf(methods, curveType=curveType, name=name, resolution=resolution)
end
function plotperf(methods; curveType="pr", name="", resolution=600)

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

    # generate the random predictors
    xmean = zeros(resolution)
    ymean = zeros(resolution)
    layers = Any[]
    truth = methods[first(keys(methods))][1]
    numRandom = 10
    for i in 1:numRandom
        predictor = rand(length(truth))
        rocData = MLBase.roc(round(Int64, truth), float(predictor), resolution)
        vals = collect(map(x->(xmap(x), -ymap(x)), rocData))
        sort!(vals)
        xvals = map(x->x[1], vals)
        yvals = map(x->-x[2], vals)
        xmean .+= xvals
        ymean .+= yvals
        aucValue = @sprintf("%0.03f", MLBasePlotting.area_under_curve(xvals, yvals))
        push!(layers, layer(
            x=xvals, y=yvals,
            Geom.line,
            Theme(default_color=color("lightgrey"))
        ))
    end
    xmean /= numRandom
    ymean /= numRandom
    push!(layers, layer(
        x=xmean, y=ymean,
        Geom.line,
        Theme(default_color=color("grey"))
    ))


    # plot the mean of the random predictors
    labels = ASCIIString[]
    xdata = Float64[]
    ydata = Float64[]
    aucValue = @sprintf("%0.03f", area_under_curve(xmean, ymean))
    append!(labels, [repeat(["AUC = $aucValue, random"], inner=[length(xmean)])])
    append!(xdata, xmean)
    append!(ydata, ymean)


    # plot the actual passed data
    for (key,(truth,predictor)) in methods
        rocData = MLBase.roc(round(Int64, truth), float(predictor), resolution)
        vals = collect(map(x->(xmap(x), -ymap(x)), rocData))
        sort!(vals)
        xvals = map(x->x[1], vals)
        yvals = map(x->-x[2], vals)
        aucValue = @sprintf("%0.03f", MLBasePlotting.area_under_curve(xvals, yvals))
        append!(labels, [repeat(["AUC = $aucValue, $key"], inner=[length(xvals)])])
        append!(xdata, xvals)
        append!(ydata, yvals)
    end

    plot(
        Guide.title("$name"),
        Guide.XLabel(xlabel),
        Guide.YLabel(ylabel),
        Guide.colorkey("Methods"),
        layer(
            x=xdata, y=ydata, color=labels,
            Geom.line
        ),
        layers...,
        Scale.discrete_color_manual(["grey", "blue", "red", "green", "purple", "pink", "orange"][1:length(methods)+1]...)
    )
end

function area_under_roc(truth::AbstractVector, predictor::AbstractVector)
    rocData = MLBase.roc(round(Int64, truth), float(predictor), resolution)
    vals = collect(map(x->(xmap(x), -ymap(x)), rocData))
    sort!(vals)
    xvals = map(x->x[1], vals)
    yvals = map(x->-x[2], vals)
    aucValue = @sprintf("%0.03f", area_under_curve(xvals, yvals))
end

# handles NaN values by using previous valid values
function area_under_curve(x, y) # must be sorted by increasing x
    area = 0.0
    lastVal = NaN
    for i in 2:length(x)
        v = (y[i-1]+y[i])/2 * (x[i]-x[i-1])
        if !isnan(v)
            area += v
            lastVal = v
        elseif !isnan(lastVal)
            area += lastVal
        end
    end
    area
end

end # module
