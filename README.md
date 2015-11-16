# MLBasePlotting

[![Travis Build Status](https://travis-ci.org/slundberg/MLBasePlotting.jl.svg?branch=master)](https://travis-ci.org/slundberg/MLBasePlotting.jl)
[![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/jt8jmirv45onel7m?svg=true)](https://ci.appveyor.com/project/slundberg/mlbaseplotting-jl)

Plotting utilities wrapper for MLBase using Gadfly.

## Installation

```julia
Pkg.clone("https://github.com/slundberg/MLBasePlotting.jl.git")
```

## Usage

```julia
using MLBasePlotting

truthValues = rand([true,false,false,false], 100)
predValues = truthValues .+ randn(100)*0.7

auc_pr = area_under_pr(truthValues, predValues)

plotperf(truthValues, predValues)
```
![Sample PR plot](/doc/samplePlotPR.png?raw=true)
