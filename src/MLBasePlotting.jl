module MLBasePlotting

export plotperf, area_under_pr, area_under_roc

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
        warn("The ROC curve type is not yet fully debugged.")
        xlabel = "False Positive Rate"
        ylabel = "True Positive Rate"
        xmap = false_positive_rate
        ymap = true_positive_rate
        if name == "" name = "ROC" end
    elseif curveType == "pr"
        xlabel = "Recall"
        ylabel = "Precision"
        xmap = recall
        ymap = precision
        if name == "" name = "Precision/Recall" end
    end

    # generate the random predictors
    xmean = zeros(resolution)
    ymean = zeros(resolution)
    layers = Any[]
    truth = methods[first(keys(methods))][1]
    numRandom = 10
    for i in 1:numRandom
        predictor = invperm(sortperm(rand(length(truth))))
        rocData = MLBase.roc(round(Int64, truth), float(predictor), resolution)
        vals = collect(map(x->(xmap(x), -ymap(x)), rocData))
        sort!(vals)
        xvals = map(x->x[1], vals)
        yvals = map(x->-x[2], vals)
        xmean .+= xvals
        ymean .+= yvals
        aucValue = @sprintf("%0.03f", MLBasePlotting.area_under_curve(xvals, yvals))
        push!(layers, layer(
            x=[0.0; xvals], y=[yvals[1]; yvals],
            Geom.line,
            Theme(default_color=colorant"lightgrey")
        ))
    end
    xmean /= numRandom
    ymean /= numRandom
    push!(layers, layer(
        x=xmean, y=ymean,
        Geom.line,
        Theme(default_color=colorant"grey")
    ))


    # plot the mean of the random predictors
    labels = ASCIIString[]
    xdata = Float64[]
    ydata = Float64[]
    aucValue = @sprintf("%0.03f", area_under_curve([0.0; xmean], [ymean[1]; ymean]))
    append!(labels, collect(repeat(["AUC = $aucValue, random"], inner=[length(xmean)+1])))
    append!(xdata, [0.0; xmean])
    append!(ydata, [ymean[1]; ymean])


    # plot the actual passed data
    for (key,(truth,predictor)) in methods
        rocData = MLBase.roc(round(Int64, truth), float(invperm(sortperm(predictor))), resolution)
        vals = collect(map(x->(xmap(x), -ymap(x)), rocData))
        sort!(vals)
        xvals = map(x->x[1], vals)
        yvals = map(x->-x[2], vals)
        aucValue = @sprintf("%0.03f", MLBasePlotting.area_under_curve([0.0; xvals], [yvals[1]; yvals]))
        append!(labels, collect(repeat(["AUC = $aucValue, $key"], inner=[length(xvals)+1])))
        append!(xdata, [0.0; xvals])
        append!(ydata, [yvals[1]; yvals])
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
        Scale.color_discrete_manual(["grey", "blue", "red", "green", "purple", "pink", "orange"][1:length(methods)+1]...)
    )
end

function area_under_pr(truth::AbstractVector, predictor::AbstractVector; resolution=4000)
    rocData = MLBase.roc(round(Int64, truth), float(invperm(sortperm(predictor))), resolution)
    vals = collect(map(x->(recall(x), -precision(x)), rocData))
    sort!(vals)
    xvals = map(x->x[1], vals)
    yvals = map(x->-x[2], vals)
    area_under_curve([0.0; xvals], [yvals[1]; yvals]) # make sure we extend all the way to zero
end
function area_under_roc(truth::AbstractVector, predictor::AbstractVector; resolution=4000)
    rocData = MLBase.roc(round(Int64, truth), float(invperm(sortperm(predictor))), resolution)
    vals = collect(map(x->(false_positive_rate(x), -true_positive_rate(x)), rocData))
    sort!(vals)
    xvals = map(x->x[1], vals)
    yvals = map(x->-x[2], vals)
    area_under_curve([0.0; xvals], [yvals[1]; yvals]) # make sure we extend all the way to zero
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
