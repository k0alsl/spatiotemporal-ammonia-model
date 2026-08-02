# Activate environment
import Pkg
Pkg.activate()
Pkg.instantiate()

# Load packages
begin
    # General
	using Dates
	using DataFrames, CSV
	# Geo
	using GeoStats
	# Stats & fitting
	using LinearAlgebra, Statistics, StatsBase
	import MultivariateStats, KissSmoothing
	# Models
	import MLJBase, MLJModels, MLJ
	import PartialLeastSquaresRegressor, MLJLinearModels
end
begin
    MLJ.@load PLSRegressor pkg=PartialLeastSquaresRegressor verbosity=0
	MLJ.@load LinearRegressor pkg=MLJLinearModels verbosity=0
end




## Cross-validation metrics ##
# Cross-validation R2
function cv_r2(pred::AbstractVector,obs::AbstractVector)::Float64
	idx = (ismissing.(pred) .== false) .* (ismissing.(obs) .== false)
    sst = sum((obs[idx] .- mean(obs[idx])).^2)
    ssr = sum((obs[idx] .- pred[idx]).^2)
    return max(0, 1 - ssr/sst)
end
function cv_r2(pred::AbstractArray,obs::AbstractArray)::Vector
	return [cv_r2(pred[:,i],obs[:,i]) for i in 1:size(pred,2)]
end

# Regression (Pearson) R with missings
function lincor(pred::AbstractVector,obs::AbstractVector)::Float64
	idx = (ismissing.(pred) .== false) .* (ismissing.(obs) .== false)
	return cor(pred[idx],obs[idx])
end
function lincor(pred::AbstractArray,obs::AbstractArray)::Vector
	return [lincor(pred[:,i],obs[:,i]) for i in 1:size(pred,2)]
end

# RMSE by column with missings
function myrmse(pred::AbstractVector,obs::AbstractVector)::Float64
	idx = (ismissing.(pred) .== false) .* (ismissing.(obs) .== false)
	return StatsBase.rmsd(disallowmissing(pred[idx]),disallowmissing(obs[idx]))
end
function myrmse(pred::AbstractArray,obs::AbstractArray)::Vector
	return [myrmse(pred[:,i],obs[:,i]) for i in 1:size(pred,2)]
end



function calc_timetrends(longterm_data::AbstractArray,number_timetrends::Int64,smooth_factor::Float64)::Matrix
    longterm_data = disallowmissing(longterm_data)
    
    # PCA
    timetrend_model = MultivariateStats.fit(MultivariateStats.PCA, longterm_data', maxoutdim=number_timetrends)
    pca_timetrends = MultivariateStats.transform(timetrend_model, longterm_data')'
    
    # Smoothing
    smooth_timetrends = zeros(size(pca_timetrends))
    for i in 1:number_timetrends
        smooth_timetrends[:,i] .= KissSmoothing.denoise(pca_timetrends[:,i], factor=smooth_factor)[1]
    end

    return smooth_timetrends
end

function regress_beta(data::AbstractArray,smooth_timetrends::AbstractArray)::Matrix
    number_timetrends = size(smooth_timetrends,2)
    nsite = size(data,2)
    coefs = zeros(number_timetrends+1, nsite)
    keepidx = ismissing.(data) .== false

    for i in 1:nsite
        x = DataFrame(smooth_timetrends[keepidx[:,i],:], :auto)
        y = disallowmissing(data[keepidx[:,i],i])
        linmod = MLJLinearModels.LinearRegressor(fit_intercept=true)
        mach = MLJBase.machine(linmod, x, y)
        MLJ.fit!(mach, verbosity=0)
        p = MLJ.fitted_params(mach)
        coefs[1,i] = p.intercept
        coefs[2:number_timetrends+1,i] .= [p.coefs[j][2] for j in 1:number_timetrends]
    end

    return coefs
end

function calc_mu(smooth_timetrends::AbstractArray, coefs::AbstractArray)
    return hcat(ones(size(smooth_timetrends,1)), smooth_timetrends) * coefs
end

function cv_ntimetrends_each(longterm_data::AbstractArray, number_timetrends::Int64, smooth_factor::Float64, ifold::Int64)::Vector
    idx = setdiff(Vector(1:size(longterm_data,2)), [ifold])
    data = longterm_data[:,idx]

    timetrends1 = calc_timetrends(data, number_timetrends, smooth_factor)
    coefs1 = regress_beta(longterm_data, timetrends1)
    preds = calc_mu(timetrends1, coefs1)[:,ifold]

    return preds
end

function cv_ntimetrends(longterm_data::AbstractArray, smooth_factor::Float64)::DataFrame
    nfolds = size(longterm_data,2)
    max_ntimetrends = size(longterm_data,2)-1
    preds = zeros(size(longterm_data)..., max_ntimetrends)
    
    for n in 1:max_ntimetrends
    for ifold in 1:nfolds
        preds[:,ifold,n] = cv_ntimetrends_each(longterm_data, n, smooth_factor, ifold)
    end
    end

    stats = DataFrame(
        nbasis = Vector(1:max_ntimetrends),
        RMSE = [myrmse(vec(preds[:,:,n]),vec(longterm_data)) for n in 1:max_ntimetrends],
        CV_R2 = [cv_r2(vec(preds[:,:,n]),vec(longterm_data)) for n in 1:max_ntimetrends],
    )

    return stats
end

function pls_regress_beta(X,X_train::AbstractArray,Y_train::AbstractArray,number_pls::Int64)::NamedTuple
    number_timetrends = size(Y_train,1)-1
    number_sites = size(X,1)

    Y_hat = zeros(number_timetrends+1,number_sites)
    plsmod = PartialLeastSquaresRegressor.PLSRegressor(n_factors=number_pls)
    mach_list = []

    for i in 1:number_timetrends+1
        plsmach = MLJBase.machine(plsmod, DataFrame(X_train, :auto), Y_train[i,:])
        MLJ.fit!(plsmach, verbosity=0)
        push!(mach_list,copy(plsmach))
        Y_hat[i,:] = MLJ.predict(plsmach,X)
    end

    return (pred=Y_hat, machines=mach_list)
end
    
function cv_npls_each(X, Y, npls, ifold)
    idx = setdiff(Vector(1:size(Y,2)), [ifold])
    X_train = X[idx,:]
    Y_train = Y[:,idx]
    preds = pls_regress_beta(X, X_train, Y_train, npls).pred[:,ifold]
    return preds
end

function cv_npls(X, Y)::DataFrame
    nfolds = size(Y,2)
    ntimetrends = size(Y,1)-1
    max_npls = nfolds
    preds = zeros(size(Y)..., max_npls)
    
    for n in 1:max_npls
    for ifold in 1:nfolds
        preds[:,ifold,n] = cv_npls_each(X, Y, n, ifold)
    end
    end

    stats = DataFrame(
        nbasis = Vector(1:max_npls),
        RMSE = [myrmse(vec(preds[:,:,n]),vec(Y)) for n in 1:max_npls],
        CV_R2 = [cv_r2(vec(preds[:,:,n]),vec(Y)) for n in 1:max_npls],
    )

    return stats
end

function krig_any(vals::AbstractArray, geometry_train::AbstractArray, geometry_test, range::Float64, sill::Float64, nugget::Float64)
    gt = DataFrame(x=vals, geometry=geometry_train) |> dropmissing |> GeoTable
    vgram = ExponentialVariogram(range=range, sill=sill, nugget=nugget)
    prob = EstimationProblem(gt, PointSet(geometry_test), :x)
    solver = KrigingSolver(:x => (variogram=vgram,))
    sol = solve(prob, solver)
    return sol.x
end

function krig_beta_resid(vals::AbstractArray, geometry_train::AbstractArray, geometry_test, range::Float64, sill::Float64, nugget::Float64)
    number_timetrends = size(vals,1) - 1
    number_sites = let
        x = length(geometry_test)
        if x < 1
            x = 1
        end
        Int(x)
    end
    results = zeros(number_timetrends+1,number_sites)
    
    for i in 1:number_timetrends+1
        results[i,:] = krig_any(vals[i,:], geometry_train, geometry_test, range, sill, nugget)
    end

    return results
end

function krig_nu(vals::AbstractArray, geometry_train::AbstractArray, geometry_test, range::Float64, sill::Float64, nugget::Float64)
    number_time = size(vals,1)
    number_sites = let
        x = length(geometry_test)
        if x < 1
            x = 1
        end
        Int(x)
    end
    results = zeros(number_time,number_sites)

    for i in 1:number_time
        results[i,:] = krig_any(vals[i,:], geometry_train, geometry_test, range, sill, nugget)
    end

    return results
end

function stmod_predict(params::NamedTuple, point::Point, predictors::AbstractArray)::Vector
    ntime, ntt = size(params.timetrends)

    coefs = zeros(ntt+1)
    for i in 1:ntt+1
        coefs[i,:] = MLJ.predict(params.pls_mach[i], reshape(predictors,1,:))
    end
    coefs .+= params.beta_means 
    coefs .+= krig_beta_resid(params.beta_resids, params.geometry, point, params.beta_krig...)

    means = calc_mu(params.timetrends, coefs)
    residuals = krig_nu(params.resids, params.geometry, point, params.nu_krig...)

    return means .+ residuals
end
