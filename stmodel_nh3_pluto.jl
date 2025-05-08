### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 02d73899-d3e6-4000-8a30-d3f967ab56e8
using GeoStats

# ╔═╡ d1b3ab98-5974-11ee-36dc-df38e5809536
begin
	using PlutoUI
	# General
	using Dates
	using DataFrames, CSV
	# Plot
	using Plots, StatsPlots
	# Stats & fitting
	using LinearAlgebra, Statistics, StatsBase, Metrics
	using Interpolations, MultivariateStats, KissSmoothing
	using MLJBase, MLJModels, MLJ
	using PartialLeastSquaresRegressor, MLJLinearModels
end;

# ╔═╡ 551d6c28-78d3-47a5-8d86-9f0a10dd74b1
VERSION

# ╔═╡ 8806e90a-b911-4564-9d38-ff399ce7609f
begin
	@load PLSRegressor pkg=PartialLeastSquaresRegressor verbosity=0
	@load LinearRegressor pkg=MLJLinearModels verbosity=0
end;

# ╔═╡ 86a18d13-6c4d-4d23-8470-790ab4866d3a
PlutoUI.TableOfContents()

# ╔═╡ 097921c2-eea4-4f24-a7a0-6473051b54f0
md"""
# Spatiotemporal modeling for NH₃ measurements

Loosely based on this paper: https://ehp.niehs.nih.gov/doi/epdf/10.1289/ehp.1408145

"""

# ╔═╡ fefc7d86-e2f6-44da-8128-0101cf1537ca
ididx = Vector(1:10)

# ╔═╡ 3e899e68-956c-4e62-9e16-4e826c5a56c5
md"""
## Input data

Monitoring data and spatial covariates are imported as data frames `nh3` and `geovars`, respectively.
"""

# ╔═╡ bb41d9f2-c41a-4f9d-af10-3a849a50cea3
nh3 = CSV.read("nh3.csv", DataFrame)

# ╔═╡ 740d509f-3fd3-44db-b1b3-2d81ecc40219
md"""
Columns of `geovars`:

- `ID` - site ID
- `lat` and `long` - latitude and longitude
- `type` - long- and short-term site ("L" and "S")
- `x` and `y` - relative easting and northing in meters
- The rest columns - spatial covariates
"""

# ╔═╡ 6a1588fd-a8e5-49a7-b224-fb625161585f
geovars = CSV.read("cov_site.csv", DataFrame)

# ╔═╡ 682f49e4-f345-443c-9077-21036029cbeb
md"""
### Concentration Data Cleaning

We convert the monitoring data to matrix and log-transform it (`nh3_mat`). The dates for which one or more monitoring data exist are also extracted for plotting and analysis (`dates`).
"""

# ╔═╡ 5c547e00-7fd1-44d1-b3aa-b22e89293d54
nh3_mat = log10.(Matrix(nh3[!, 2:end]))

# ╔═╡ 41bb5af2-131f-4f6d-ba55-cc9e23c22006
dates = Dates.datetime2unix.(DateTime.(nh3.Date))

# ╔═╡ f6bdea65-f1cf-4c1c-9744-f75c8b9eedc0
md"Look at data values and missing data."

# ╔═╡ ac6ce0d4-d91c-46c2-82fb-dcc88a486b85
heatmap(nh3_mat; 
	xticks=(1:10, names(nh3)[2:end]),
	yticks=(1:size(nh3,1), nh3.Date),
	size=(800, 800),
	title="NH3 concentration (log-transformed)"
)

# ╔═╡ df9c40bf-7e3d-480f-8ad2-7453b726068e
md"""
The model we will create assumes a regular sample duration and no missing data. There are probably different ways we can deal with this, but we'll go with linear interpolation. `nh3_itp` and `nh3_itp_noextrap` are matrices of monitoring data with and without extrapolation, respectively, imputed on a biweekly basis--the dates are stored in `biweekly`. Extrapolation fills all missing values out of each sampling duration with mean of collected values. The extrapolated version is needed for a few functions which does not allow missing values as their arguments.
"""

# ╔═╡ c165f4fe-9422-46c7-bbd5-14786a790fb2
begin
	biweekly = collect(dates[1]:3600.0*24*14:dates[end])
	
	nh3_itp = zeros(length(biweekly), size(nh3_mat, 2))
	nh3_itp_noextrap = zeros(length(biweekly), size(nh3_mat, 2))
	
	for i ∈ eachindex(nh3_mat[1, :])
		
		present = ismissing.(nh3_mat[:, i]) .== false

		lta = mean(skipmissing(nh3_mat[:,i]))
		
		itp1 = linear_interpolation(dates[present], nh3_mat[present, i],
			extrapolation_bc=lta)

		itp2 = linear_interpolation(dates[present], nh3_mat[present, i],
			extrapolation_bc=NaN)
		
		nh3_itp[:, i] .= itp1.(biweekly)
		nh3_itp_noextrap[:, i] .= itp2.(biweekly)
	end

	nh3_itp0 = nh3_itp
	nh3_itp = nh3_itp[:,ididx]
	nh3_itp_noextrap0 = nh3_itp_noextrap
	nh3_itp_noextrap = nh3_itp_noextrap[:,ididx]
end

# ╔═╡ b45ad166-8690-49fe-a29a-af55773d25e5
begin
	labels = permutedims(names(nh3)[2:end][ididx])
	slabels = ["L1","L2","L3","S1","S2","S3","S4","S5","S6","N"]
	itp_time_ticks = (1:size(nh3_itp,1), Date.(Dates.unix2datetime.(biweekly)))
	timeplot = plot(nh3_itp,
		label=false, #labels,
		xticks=itp_time_ticks,
		xrotation=90,
		title="Interpolated data with extrapolation",
	)
end

# ╔═╡ 13e650f3-1186-442e-84cc-33900e42936a
timeplot_noextrap = plot(nh3_itp_noextrap,
	label=false, #labels,
	xticks=itp_time_ticks,
	xrotation=90,
	title="Interpolated data with no extrapolation",
)

# ╔═╡ 09d0e599-56f1-4ef2-992e-92aa2df91022
md"Let's assign variables to be frequently used--`ntime`, `nsite`, and `keepidx`."

# ╔═╡ 5fb05902-071e-4fee-9a35-021d16f20120
ntime, nsite = size(nh3_itp)

# ╔═╡ 2ce59e52-a7b5-4901-8bcc-fd3b3484be32
keepidx = isnan.(nh3_itp_noextrap) .== false

# ╔═╡ d32d0503-66b5-42db-9609-e2fd181d1132
md"""
### Geographic covariate data cleaning

First, reorder the rows of the geographic covariates to match the columns of the measurements.
"""

# ╔═╡ eb1e84e5-81f8-4fff-ac75-a05111e81e26
begin 
	order = [findfirst(geovars.ID .== x) for x in names(nh3)[2:end]]
	notmissing = (isnothing.(order)) .== false
	order = order[notmissing]
	@assert length(order) == size(nh3_mat, 2) "Sizes don't match"
	geovars_ordered1 = geovars[order, :]
end

# ╔═╡ 54d5d99f-4097-42e7-95bb-fc7193451ccf
md"Let's look at the covariance of the first ten variables."

# ╔═╡ eaa8ab2b-dc56-47ab-9eab-622a83c54f5e
md"""
Now, let's convert it to a matrix for analysis and apply tranformation for some covariates with high skewness. The consequential covariates (the names are listed in `geovars_names`) are stored as a matrix (`cov_mat`) and a data frame (`geovars_ordered`).
"""

# ╔═╡ 88f880dc-953d-4d46-a9f7-a315cff5704a
md"Skewness threshold to transform data above:"

# ╔═╡ 4ed1da32-3cfb-4ef3-9cd9-87cb0f0e2f66
@bind skew_threshold Slider(0.1:0.1:2.0, default=0.5, show_value=true)

# ╔═╡ b37e0dc1-2481-41d2-a923-170af872090d
cov_mat = let
	x = Matrix(geovars_ordered1[ididx, Not("ID", "lat", "long", "x", "y", "type")])
	skew = [skewness(x[:,i]) for i in 1:size(x,2)]
	for i in 1:size(x,2)
		if skew[i] > skew_threshold
			x[:,i] = log10.(x[:,i] .+ .1)
		elseif skew[i] < -skew_threshold
			x[:,i] = -log10.(1000. .- x[:,i])
		end
	end
	x
end

# ╔═╡ 94f41856-fc33-4615-83f7-558888f13f14
geovars_ordered = let
	x1 = geovars_ordered1[ididx,["ID", "lat", "long", "x", "y", "type"]]
	x = DataFrame(cov_mat, names(geovars_ordered1[ididx, Not("ID", "lat", "long", "x", "y", "type")]))
	hcat(x1,x)
end

# ╔═╡ 96d7d71a-c121-4066-9f7c-3815459461f0
ngeovars = ncol(geovars_ordered)-6

# ╔═╡ bdafc38a-dbdf-46a8-95eb-cc72fa937e3a
let
	x = geovars_ordered1[!, Not("ID", "lat", "long", "x", "y", "type")]
	p = [histogram(x[:,i], nbin=10, title=names(x)[i], titlefontsize=7) for i in 1:ngeovars]
	plot(p..., legend=false,
		layout=(:,8), size=(1000,1400))
end

# ╔═╡ 6fe4f6d3-ec1b-4b2a-8bea-6b881388382a
geovars_names = names(geovars_ordered[!, Not("ID", "lat", "long", "x", "y", "type")])

# ╔═╡ 4c9e6812-a3fa-4815-ab0e-96ef54b64fe5
md"Let's check the histograms again."

# ╔═╡ bf49745e-3fc1-4a40-9a24-a0495b2c8ec3
let
	p = [histogram(cov_mat[:,i], nbin=10, title=geovars_names[i], titlefontsize=7) 
			for i in 1:size(cov_mat,2)]
	plot(p..., legend=false,
		layout=(:,8), size=(1000,1400))
end

# ╔═╡ 3e3380bf-9a34-4538-b177-d661dedbbb35
md"### *f*: Sampling data for cross-validation"

# ╔═╡ 8ae9860f-0e38-4841-a0be-2f46f050844d
function subsample_data(c_itp::Matrix, c_itp_noexp::Matrix, cov::DataFrame, nums::Vector)
	c_itp = c_itp[:,nums]
	c_itp_noexp = c_itp_noexp[:,nums]
	cov = DataFrame(cov[nums,:])
	cov_mat = Matrix(cov[!, Not("ID", "lat", "long", "x", "y", "type")])
	return (c_itp=c_itp, c_itp_noexp=c_itp_noexp, 
			cov=cov, cov_mat=cov_mat,
			)
end

# ╔═╡ 478ee350-1092-475e-acd9-96f49faa2a6a
function c_inv_norm(c, c_means, c_stds)
	c .* c_stds .+ c_means
end

# ╔═╡ d7458523-5253-4526-9b18-ea50373f2f79
md"""
## Time trends
Now, let's create temporal basis functions. We'll only use the long-term measurements (type=="L") for the time trends (`nh3_longterm`).
"""

# ╔═╡ 53d1ce27-4189-4d4b-9ec7-247530a88a3f
nh3_longterm = nh3_itp[:, geovars_ordered.type .== "L"]

# ╔═╡ bbcef19e-edbd-4f78-ba89-d0128c99922d
let 
	x = geovars_ordered.type .== "L"
	heatmap(nh3_longterm, yticks=itp_time_ticks, xticks=(1:sum(x), labels[x]))
end

# ╔═╡ 49176761-5498-408a-831f-b6832651d517
md"""
### PCA for time trends

There are many ways we could make the trends (`timetrends`), but we will use PCA with the number of components:
"""

# ╔═╡ 648a7ba6-be2c-4e98-8b54-41f13380448d
@bind num_timetrends Slider(1:size(nh3_longterm,2), show_value=true, default=2)

# ╔═╡ 306231bd-f42f-4a03-96dc-c994e1279084
timetrend_model = MultivariateStats.fit(MultivariateStats.PCA, nh3_longterm', maxoutdim=num_timetrends)

# ╔═╡ 0971cb1e-b038-4c95-bf1a-98d84b3c04c9
begin
	timetrends = MultivariateStats.transform(timetrend_model, nh3_longterm')'

	# Fitting
	beta_l = zeros(num_timetrends,size(nh3_longterm,2))
	for i ∈ eachindex(nh3_longterm[1,:])
		idx = isnan.(nh3_itp_noextrap[:, geovars_ordered.type .== "L"]) .== false
		idx = idx[:,i]
		x = DataFrame(timetrends[idx,:], :auto)
		y = nh3_longterm[idx, i]
		linmod = MLJLinearModels.LinearRegressor(fit_intercept=true)
		mach = machine(linmod, x, y)
		MLJ.fit!(mach, verbosity=0)
		p = fitted_params(mach)
		beta_l[:, i] .= [p.coefs[i][2] for i in 1:length(p.coefs)]
	end

	# Inverse time trend if fitted coefficients are negative on average
	for i in eachindex(timetrends[1, :])
		if mean(beta_l[i,:]) < 0
			timetrends[:, i] *= -1
		end
	end
	
	timetrends
end

# ╔═╡ dc5b4a8c-3035-415c-ab69-60c987ab5200
md"""
This is what the time trends (right) look like next to the original data (left):
"""

# ╔═╡ d96fcb89-3193-4bd5-9cf2-f0d90bf70e89
plot(timeplot_noextrap, 
	plot(timetrends, label=reshape(["f$i" for i in 1:num_timetrends], (1,:)),
		xticks=itp_time_ticks, x_rotation=90, title="Primary time trends by PCA"), 
	size=(800, 400))

# ╔═╡ 5795c94d-ed24-40b5-9bf2-bdda3adfe7a4
md"""
### Smoothing time trends

To reduce overfitting, we will smooth our time trends to produce `smooth_timetrends`.

We need to choose a "smoothing intensity": https://github.com/francescoalemanno/KissSmoothing.jl

"""

# ╔═╡ 43da7ff6-34c3-4aa3-bcf9-ca1536ba22a5
@bind smooth_intensity Slider(0:0.1:1.5, default=0.5, show_value=true)

# ╔═╡ 1200206c-a7af-423c-b655-6fb7c2e8e0b6
begin
	smooth_timetrends = zeros(ntime,num_timetrends)
	for i in eachindex(timetrends[1, :])
		smooth_timetrends[:, i] .= denoise(timetrends[:, i], factor=smooth_intensity)[1]
	end

	smooth_timetrends
end

# ╔═╡ 8bdb179f-a565-429c-91fa-f3222286563d
md"""
This is what the smoothed time trends look like along with the original time trends:
"""

# ╔═╡ 66b67223-02fc-4cd3-836b-fbd36e9dfab4
let
	p = plot(timetrends, ls=:dash, 
		label=reshape(["f$i" for i in 1:num_timetrends], (1,:)),
	)
	plot!(smooth_timetrends, lc=collect(1:num_timetrends)',
		label=reshape(["f$(i)-smooth" for i in 1:num_timetrends], (1,:)), 
		xticks=itp_time_ticks, x_rotation=90, bottom_margin=5Plots.mm,
		title="Temporal basis functions",
	)
end

# ╔═╡ 907e5cdb-a332-4b77-9538-b10374c0d3e6
md"""
### Fit to time trends
We first fit our measurement data to the time trends to get predictions of coefficients (`β`) at each location.
"""

# ╔═╡ a46e8775-f17a-4fe2-a3a6-85d50f80dcf9
begin
	timetrend_w = zeros(num_timetrends, nsite)
	timetrend_b = zeros(nsite)
	for i ∈ eachindex(nh3_itp[1,:])
		idx = isnan.(nh3_itp_noextrap[:, i]) .== false
		x = DataFrame(smooth_timetrends[idx,:], :auto)
		y = nh3_itp_noextrap[idx, i]
		linmod = MLJLinearModels.LinearRegressor(fit_intercept=true)
		mach = machine(linmod, x, y)
		MLJ.fit!(mach, verbosity=0)
		p = fitted_params(mach)
		timetrend_w[:, i] .= [p.coefs[i][2] for i in 1:length(p.coefs)]
		timetrend_b[i] = p.intercept
	end
end

# ╔═╡ 0224a5a1-839f-4a15-9401-e029a10004d7
β = vcat(timetrend_b',timetrend_w)

# ╔═╡ fa24dcb0-deaf-4cbd-bfe0-7e08bb3152c5
scatter(β[2:end,:]', 
	xticks=(1:nsite,slabels), 
	label=reshape(["β$i" for i in 1:num_timetrends], (1,:))
)

# ╔═╡ b47aff02-2b0c-4861-8551-80773adace56
timetrend_pred = smooth_timetrends * timetrend_w .+ timetrend_b'

# ╔═╡ 1307d582-e272-4aea-bca7-101dc8797d6a
md"Let's check if we're making an okay prediction."

# ╔═╡ ac379534-07e2-40d5-b4f8-cd9582113d0c
let
	p_total = []
	for i in 1:nsite
		p1 = scatter(nh3_itp_noextrap[:,i], xticks=false, #xticks=itp_time_ticks, x_rotation=90, 
			label = "Obs", legend_position=:topright, 
			title=labels[i])
		plot!(timetrend_pred[:,i], ls=:dash, label="Fit")
		push!(p_total, p1)
	end
	plot(p_total..., layouts=(:,2), size=(900,900), plot_title="Time trend fit with linear regression")
end

# ╔═╡ 91681a1e-c0ee-4dfc-b6fa-d5847ee3db22
md"### *f*: Calculate Time trends"

# ╔═╡ 69c2b05a-43b5-4db0-902a-0b64121620f5
function regress_beta(c_itp_noextrap, tt::Matrix)::Matrix
	coef = zeros(size(tt,2), size(c_itp_noextrap,2))
	intercept = zeros(size(c_itp_noextrap,2))
	for i ∈ eachindex(c_itp_noextrap[1,:])
		idx = isnan.(c_itp_noextrap[:, i]) .== false
		x = DataFrame(tt[idx,:], :auto)
		y = c_itp_noextrap[idx, i]
		linmod = MLJLinearModels.LinearRegressor(fit_intercept=true)
		mach = machine(linmod, x, y)
		MLJ.fit!(mach, verbosity=0)
		p = fitted_params(mach)
		coef[:, i] .= [p.coefs[i][2] for i in 1:length(p.coefs)]
		intercept[i] = p.intercept
	end
	return vcat(intercept',coef)
end

# ╔═╡ b8768516-94b8-4727-bc9c-93da5ae9fc48
md"""
## PLS covariates

The next step is to compress the geographic covariate information using PLS scores.
"""

# ╔═╡ b2a0739d-e21c-4a9b-a285-fb38e4c076fe
md"""
### Standardize covariates
First, we need to standardize geographic covariates.
"""

# ╔═╡ 04650a44-3062-4610-9638-2b36f3404c33
begin
	std_machine = machine(Standardizer(), DataFrame(cov_mat, :auto))
	MLJ.fit!(std_machine)
	cov_std = Matrix(MLJ.transform(std_machine, DataFrame(cov_mat, :auto)))
end

# ╔═╡ be91c45c-6e58-4fc0-b05d-bfd40d68b30a
md"""
### Calculate PLS scores

Now, let's create our PLS model.
"""

# ╔═╡ 34bac7af-961e-446b-84fa-11c2fd955b28
md"The number of PLS components:"

# ╔═╡ 4d92f4aa-dabf-4612-8f7e-ee94b4a4ff42
@bind num_pls_factors Slider(1:10, show_value=true, default=2)

# ╔═╡ 27398978-0a4f-4a5e-b6a8-59c6baa1bbd5
md"""
Let's look into our PLS model. PLS scores, `pls_scores`, which go into the model, are calculated by multiplying standardized covariates and PLS parameter *W*.
"""

# ╔═╡ 23f9a95b-e931-44b6-8c00-81f348345f27
begin	
	pls_scores = zeros(nsite, num_pls_factors, num_timetrends+1)
	pls_w = zeros(size(cov_std,2), num_pls_factors, num_timetrends+1)
	pls_p = zeros(size(cov_std,2), num_pls_factors, num_timetrends+1)
	pls_model = PartialLeastSquaresRegressor.PLSRegressor(n_factors=num_pls_factors)
	for i in 1:num_timetrends+1
		pls_machine = machine(pls_model, DataFrame(cov_std, :auto), 
			β[i,:])
		MLJ.fit!(pls_machine, verbosity=0)
		params = fitted_params(pls_machine)
		pls_w[:,:,i] = params.fitresult.W
		pls_scores[:,:,i] = cov_std * params.fitresult.W
		pls_p[:,:,i] = params.fitresult.P
	end
end

# ╔═╡ 6a106804-0cea-492c-b502-7fcfba09bbb9
pls_scores[:,:,1]

# ╔═╡ 11280080-e031-44da-a53b-ff2756359fe9
md"""
### Read PLS loadings

PLS loadings, `pls_p`, are not used in creating our model, but tell us about relative significances of covariates.
"""

# ╔═╡ 4029a2ad-5a7a-4632-a1b9-28f6c71c6ecc
let
	p = heatmap(pls_p[:,1,:], cmap=cgrad(:vik, 11, categorical=true),
		xticks=(1:num_timetrends+1, ["β$(i-1)" for i in 1:num_timetrends+1]),
		yticks=(1:size(cov_std,2), geovars_names), yflip=true, 
		size=(450,800), left_margin=20Plots.mm, tickfontsize=7,
		title="PLS loadings (1st components)"
	)
	# savefig(p, "Figs/pls_loadings_abs_invert.pdf")
end

# ╔═╡ fa0c624a-d3fe-4a6a-83ed-f9a72f81dc4d
# PLS loadings
let
	lim = maximum(abs.(pls_p)) + .05
	p = [scatter(pls_p[:,:,i], title="β$(i-1)",
		xticks=1:size(cov_std,2), xrotation=90, # xticks=(1:75, geovars_names), 
		ylims=(-lim,lim),
		label=collect(1:6)', legendcolumns=num_pls_factors, # legend=:outertop, 
		palette=palette([:red, :orange, :white], num_pls_factors),
		# size=(1000,550), bottom_margin=23Plots.mm)
		) for i in 1:num_timetrends+1]
	plot(p..., layout=(num_timetrends+1,1), size=(1000,900), plot_title="PLS loadings (all components)")
end

# ╔═╡ 286067f0-61f9-4781-8217-39dc9f994895
md"### *f*: Standardize data and calculate PLS scores"

# ╔═╡ 4c87672f-8038-48c2-90bd-0a53ef7a40eb
# For model
function standardize_data(β::Matrix, cov_mat::Matrix)
	β_means = mean(β, dims=2)
	std_machine = machine(Standardizer(), DataFrame(cov_mat, :auto))
	MLJ.fit!(std_machine)
	cov_std = Matrix(MLJ.transform(std_machine, DataFrame(cov_mat, :auto)))
	return (β_means=β_means, cov_std=cov_std, std_machine=std_machine)
end

# ╔═╡ e3e5dcce-78cf-4b77-b864-231742c5ef72
# For estimation
function standardize_data(cov_mat::Matrix, std_machine::Machine)::Matrix
	cov_std = Matrix(MLJ.transform(std_machine, DataFrame(cov_mat, :auto)))
end

# ╔═╡ fd8c02d7-1c37-4410-a26e-596abe95130f
# For model
function calc_pls(cov_std::Matrix, β::Matrix, n_pls::Int64)
	pls_scores = zeros(size(cov_std,1), n_pls, size(β,1))
	pls_w = zeros(size(cov_std,2), n_pls, size(β,1))
	pls_model = PartialLeastSquaresRegressor.PLSRegressor(n_factors=n_pls)
	for i in 1:size(β,1)
		pls_machine = machine(pls_model, DataFrame(cov_std, :auto), 
			β[i,:])
		MLJ.fit!(pls_machine, verbosity=0)
		params = fitted_params(pls_machine)
		pls_w[:,:,i] = params.fitresult.W
		pls_scores[:,:,i] = cov_std * pls_w[:,:,i]
	end
	return (pls_scores=pls_scores, pls_w=pls_w)
end

# ╔═╡ 73103277-3434-44c7-b513-1ac4ef5beb50
# For estimation
function calc_pls(cov_std::Matrix, pls_w::Array, n_tt::Int64)::Array
	pls_scores = zeros(size(cov_std,1), size(pls_w,2), n_tt+1)
	for i in 1:n_tt+1
		pls_scores[:,:,i] = cov_std * pls_w[:,:,i]
	end
	return pls_scores
end

# ╔═╡ 5e69fe99-1ea4-4aa8-847c-8e1e273639d5
md"""
## Time trend coefficients

### Time trend coefficient fitting
Let's fit the time trend coefficients (`β` fields) based on our PLS scores. In this step,

- predictor variables are PLS scores (`pls_scores`); and
- dependent variables are means of time trend coefficients (true values in `β`, and predicted values in `β̄`).
"""

# ╔═╡ 37213b75-8ca2-40a5-b8f5-056a86d08a2c
begin
	α = zeros(num_pls_factors, num_timetrends+1)
	for i in 1:num_timetrends+1
		α[:,i] = let
			xx = DataFrame(pls_scores[:,:,i], :auto)
			yy = β[i,:] 
			pls_linmod = MLJLinearModels.LinearRegressor(fit_intercept=false)
			mach = machine(pls_linmod, xx, yy)
			MLJ.fit!(mach, verbosity=0)
			p = fitted_params(mach)
			[x[2] for x in p.coefs]
		end
	end
end

# ╔═╡ 5d7e353b-e9f2-4378-a970-846fb9a65c9e
begin
	β̄ = zeros(num_timetrends+1, nsite)
	p1 = []
	for i in 1:num_timetrends+1
		β̄[i,:] = pls_scores[:,:,i]*α[:,i] .+ mean(β[i,:])
		p11 = scatter(β[i,:], β̄[i,:],
			title="β$(i-1)", legend=false,
			xlab="Fitted coef", ylab="Predicted coef")
		Plots.abline!(1, 0, c=:black, line=:dash)
		push!(p1,p11)
	end
	p_pls = plot(p1..., layouts=(:,3), size=(900,300),
		left_margin=4Plots.mm, bottom_margin=4Plots.mm)
end

# ╔═╡ a2f0a3e9-b9e3-41d5-bf88-84c0eaae3dd0
let
	print("R² values for βs prediction: ")
	print([round(Metrics.r2_score(β̄[i,:], β[i,:]); digits=4) for i in 1:num_timetrends+1])
end

# ╔═╡ 7f24b41a-9e8d-4313-81b0-552e3ebc13d0
let
	p_total = []
	tt1 = hcat(ones(ntime), smooth_timetrends) * β̄
	for i in 1:nsite
		p1 = scatter(nh3_itp_noextrap[:,i], xticks=false, #xticks=itp_time_ticks, x_rotation=90, 
			label = "Obs", legend_position=:topright, 
			title=labels[i])
		plot!(tt1[:,i], ls=:dash, label="Pred")
		push!(p_total, p1)
	end
	p = plot(p_total..., layouts=(:,2), size=(900,900), plot_title="Time trend fits with PLS scores")
end

# ╔═╡ 76997d3a-981f-4691-8c44-06e4956d2ee7
md"""
### Time trend coefficient kriging

Let's add some kriging to account for spatial autocorrelation in the β fields.

Here, we're doing universal kriging, which means we first fit a deterministic model to the data (which we did with PLS above), then we fit a kriging model to the residuals of the deterministic model fit. The resulting model will fit the training data exactly and will attempt to account for spatial autocorrelation in parameters when making predictions at locations not included in the training data.
"""

# ╔═╡ f1c123b5-6c59-414e-aff1-66aeb660e365
begin
	β_resid = β .- β̄
	plot(β_resid', xticks=(1:nsite, labels), x_rotation=90, 
	label=reshape(["β$(i-1)" for i in 1:num_timetrends+1], (1,:)), 
	bottom_margin=4Plots.mm,
	title="Residuals in βs")
end

# ╔═╡ fb06a012-29c0-4dfa-bfca-6f8d8b820efc
md"Let's set kriging parameters (range, sill, and nugget) here:"

# ╔═╡ b728bf43-6853-417d-a705-f24849994b60
β_range, β_sill, β_nugget=1100.0, 0.035, 0.019

# ╔═╡ 424c7e26-83c2-4042-9529-619109d51c8d
begin
	β_names = ["β$(i-1)" for i in 1:num_timetrends+1]
	β_df = DataFrame(zeros(nsite, num_timetrends+1), β_names)
	β_var_df = DataFrame(zeros(nsite, num_timetrends+1), β_names)
	
	β_geo = georef(DataFrame(β_resid', β_names),
			PointSet(Matrix(geovars_ordered[:, [:x, :y]])'))
	β_vgram = ExponentialVariogram(range=β_range, sill=β_sill, nugget=β_nugget)
	
	for i in 1:num_timetrends+1
		β_problem = EstimationProblem(β_geo, β_geo.geometry, Symbol(β_names[i]))
		β_solver = KrigingSolver(Symbol(β_names[i]) => (variogram=β_vgram,)) #mean=0.0))
		β_sol = solve(β_problem, β_solver)
		β_df[:,i] = β_sol[:,β_names[i]]
		β_var_df[:,i] = β_sol[:,β_names[i]]
	end
	
	β̂ = β̄ .+ Matrix(β_df)'
end

# ╔═╡ 94e3091f-7ccc-4878-acbe-b7e93737b838
md"We can try fitting empirical variogram to theoretical variogram (e.g. Exponential, Gaussian) to estimate kriging parameters. The Exponential variogram is adopted here."

# ╔═╡ 6c1eec73-3e56-4a99-aca5-67b764be4f70
β_ranges, β_sills, β_nuggets = let
	ranges = zeros(num_timetrends+1)
	sills = zeros(num_timetrends+1)
	nuggets = zeros(num_timetrends+1)
	for i in 1:num_timetrends+1
		vgram = StatsBase.fit(ExponentialVariogram, 
			EmpiricalVariogram(β_geo, Symbol(β_names[i])))
		ranges[i] = range(vgram)
		sills[i] = sill(vgram)
		nuggets[i] = nugget(vgram)
	end
	ranges, sills, nuggets
end

# ╔═╡ ddce6b65-4439-4420-9bea-fe515fc57dde
mean(β_ranges), mean(β_sills), mean(β_nuggets)

# ╔═╡ 3fd55819-5b7f-4cb7-a459-83e700ffa5b0
let
	p_total = []
	tt1 = hcat(ones(ntime), smooth_timetrends) * β̂
	# tt1_var = abs.(hcat(ones(ntime), smooth_timetrends) * Matrix(β_var_df)')
	for i in 1:nsite
		p1 = scatter(nh3_itp_noextrap[:,i], xticks=false, 
			#xticks=itp_time_ticks, x_rotation=90, 
			label = "Obs", legend_position=:topright, 
			title=labels[i])
		plot!(tt1[:,i], label="Mean", # ribbon=tt1_var[:,i], fillalpha=0.25,  
				ls=:dash, c=2)
		push!(p_total, p1)
	end
	p = plot(p_total..., layouts=(:,2), size=(900,900), #legend_column=3,
		plot_title="Time trend fits with β kriging")
	# savefig(p, "Output/site_mean.pdf")
end

# ╔═╡ fdb7d310-51df-4133-85bd-92eb410122ed
md"### *f*: Estimate mean fields"

# ╔═╡ 480344fd-e02a-4dc1-814f-a113c5743c0e
function regress_alpha(pls_scores::Array, β::Matrix)::Matrix
	α = zeros(size(pls_scores,2), size(β,1))
	for i in 1:size(β,1)
		α[:,i] = let
			xx = DataFrame(pls_scores[:,:,i], :auto)
			yy = β[i,:] 
			pls_linmod = MLJLinearModels.LinearRegressor(fit_intercept=false)
			mach = machine(pls_linmod, xx, yy)
			MLJ.fit!(mach, verbosity=0)
			p = fitted_params(mach)
			[x[2] for x in p.coefs]
		end
	end
	return α
end

# ╔═╡ 88abe548-b780-488b-96c1-4201a50d5808
function calc_beta_means(α::Matrix, pls_scores::Array, β_means::Matrix)::Matrix
	β̄ = zeros(length(β_means), size(pls_scores,1))
	for i in 1:length(β_means)
		β̄[i,:] = pls_scores[:,:,i]*α[:,i] .+ β_means[i]
	end
	return β̄
end

# ╔═╡ 7bced889-0373-4ad1-b433-cc8bd1ff0a00
# For model
function krig_beta(β::Matrix, β̄::Matrix, cov::DataFrame, range::Float64, sill::Float64, nugget::Float64)
	
	β_resid = β .- β̄

	β_names = ["β$(i-1)" for i in 1:size(β,1)]
	β_resid_df = DataFrame(zeros(size(β')), β_names)
	β_var_df = DataFrame(zeros(size(β')), β_names)
	β_geo = georef(DataFrame(β_resid', β_names),
				PointSet(Matrix(cov[:, [:x, :y]])'))
	vgram = ExponentialVariogram(range=range, sill=sill, nugget=nugget)
	
	for i in 1:size(β,1)
		β_problem = EstimationProblem(β_geo, β_geo.geometry, Symbol(β_names[i]))
		β_solver = KrigingSolver(Symbol(β_names[i]) => (variogram=vgram,)) #mean=0.0))
		β_sol = solve(β_problem, β_solver)
		β_resid_df[:,i] = β_sol[:,β_names[i]]
		β_var_df[:,i] = β_sol[:,β_names[i]]
	end

	β_resid = Matrix(β_resid_df)'
	β_var = Matrix(β_var_df)'
	β̂ = β̄ .+ β_resid
	
	return (β_resid=β_resid, β_var=β_var, β̂=β̂)
end

# ╔═╡ 9d5313c0-9bfa-4b9f-8020-3b2b9df750ab
# For estimation
function krig_beta(β::Matrix, β̄::Matrix, cov::DataFrame, β̄_new::Matrix, cov_new::DataFrame, range::Float64, sill::Float64, nugget::Float64)
	
	β_resid = β .- β̄

	β_names = ["β$(i-1)" for i in 1:size(β,1)]
	β_resid_new_df = DataFrame(zeros(size(β̄_new')), :auto)
	β_var_new_df = DataFrame(zeros(size(β̄_new')), :auto)
	β_geo = georef(DataFrame(β_resid', β_names),
				PointSet(Matrix(cov[:, [:x, :y]])'))
	vgram = ExponentialVariogram(range=range, sill=sill, nugget=nugget)
	
	for i in 1:size(β,1)
		β_problem_new = EstimationProblem(β_geo, 
			PointSet(Matrix(cov_new[:, [:x, :y]])'), Symbol(β_names[i]))
		β_solver = KrigingSolver(Symbol(β_names[i]) => (variogram=vgram,)) # mean=0.0))
		β_sol = solve(β_problem_new, β_solver)
		β_resid_new_df[:,i] = β_sol[:,β_names[i]]
		β_var_new_df[:,i] = β_sol[:,β_names[i]]
	end

	β_resid_new = Matrix(β_resid_new_df)'
	β_var_new = Matrix(β_var_new_df)'
	β̂_new = β̄_new .+ β_resid_new
	
	return (β_resid=β_resid_new, β_var=β_var_new, β̂=β̂_new)
end

# ╔═╡ 21ed5ee0-6608-4153-9d58-06e2bcd57a76
function calc_mu(tt::Matrix, β̂::Matrix, β_var::Matrix)
	μ = hcat(ones(size(tt,1)), tt) * β̂
	μ_var = abs.(hcat(ones(size(tt,1)), tt) * β_var)
	return (μ=μ, μ_var=μ_var)
end

# ╔═╡ 0b7d3b14-cc64-401f-908a-f426f9f517fc
function calc_mu(tt::Matrix, β̂::Matrix)
	μ = hcat(ones(size(tt,1)), tt) * β̂
	return μ
end

# ╔═╡ b49b10a8-b9da-4f0a-be9c-e29fb34b719c
md"""
## Residual field
### Residual field kriging

The final step is to use kriging to account for spatial autocorrelation in the temporal trend fit residuals. 

First, we calculate the mean (`μ̂`) and residual fields (`ν`).
"""

# ╔═╡ 4ea7ab16-e830-4c6d-bda8-4e9f99f353ae
begin
	μ̂ = hcat(ones(ntime), smooth_timetrends) * β̂
	ν = nh3_itp_noextrap .- μ̂
	ν = replace(ν, NaN=>missing)
end;

# ╔═╡ 54dca345-1615-465b-92b3-f268eb590675
md"""
We're assuming that there is no temporal autocorrelation in the residuals, but we might as well check that assumption before we start:
"""

# ╔═╡ 2715625a-a5d5-47af-b454-24c9fe929a31
plot(ν, xticks=itp_time_ticks, x_rotation=90, title="Residual fields", label=labels, legend_position=:outertop, legend_column=4, bottom_margin=4Plots.mm)

# ╔═╡ bd82c75e-096c-415f-9e6a-62e8b7a3b015
md"""
Let's plot those residuals, along with the mean at each location.
"""

# ╔═╡ 237ab62e-c3c4-49ec-91b5-39fb7fbb08bc
begin
	scatter(ν', mc=:white,
		xticks=(1:nsite, labels), xrotation=90, bottom_margin=4Plots.mm,
		label=false,
	)
	scatter!([mean(skipmissing(ν[:,i])) for i in 1:nsite], 
		label=false,
	)
end

# ╔═╡ acd91af5-b39c-47d4-a02d-594b611ec51f
md"Now, we fit a kriging model with following parameters to these residuals:"

# ╔═╡ 57a91282-05ba-4165-a2e0-a7461e836610
ν_range, ν_sill, ν_nugget = 5600.0, 0.0018, 0.00068

# ╔═╡ a0dd3b4c-335a-4d24-a28d-1f00dec7e4db
begin
	ν_names = ["x$i" for i in 1:ntime]
	ν_geo = georef(DataFrame(ν', :auto), 
		PointSet(Matrix(geovars_ordered[:, [:x, :y]])'))
end

# ╔═╡ da26eacf-11af-4e2e-be8d-9e8e5730afad
begin
	ν̂_df = DataFrame(zeros(nsite,ntime), :auto)
	ν̂_var_df = DataFrame(zeros(nsite,ntime), :auto)
	ν_vgram = ExponentialVariogram(range=ν_range, sill=ν_sill, nugget=ν_nugget)
	
	for i in 1:ntime
		ν_idx1 = ismissing.(ν[i,:]) .== false
		ν_geo1 = georef(DataFrame(ν=ν[i,:][ν_idx1]),
			PointSet((Matrix(geovars_ordered[:, [:x, :y]])[ν_idx1,:])'))
		ν_problem1 = EstimationProblem(ν_geo1, ν_geo.geometry, :ν)
		ν_solver1 = KrigingSolver(:ν => (variogram=ν_vgram, mean=0.0))
		ν_sol1 = solve(ν_problem1, ν_solver1)
		ν̂_df[:,i] = ν_sol1.ν
		ν̂_var_df[:,i] = ν_sol1.ν_variance
	end
end

# ╔═╡ db622d9c-9508-4c28-9265-8182a6f1c246
md"We can try fitting empirical variogram to theoretical variogram (e.g. Exponential, Gaussian) to estimate kriging parameters. The Exponential variogram is adopted here."

# ╔═╡ aaec4bc0-27e9-4f34-bdfc-b4cdcfe61353
ν_ranges, ν_sills, ν_nuggets = let
	ranges = []
	sills = []
	nuggets = []
	for i in 1:length(ν_names)
		try
			vgram = StatsBase.fit(ExponentialVariogram, 
				EmpiricalVariogram(ν_geo, Symbol(ν_names[i])))
			push!(ranges, range(vgram))
			push!(sills, sill(vgram))
			push!(nuggets, nugget(vgram))
		catch
			push!(ranges, missing)
			push!(sills, missing)
			push!(nuggets, missing)
		end
	end
	ranges, sills, nuggets
end

# ╔═╡ c3c4cd2e-d03e-4294-86b2-3f9960777673
mean(skipmissing(ν_ranges)), mean(skipmissing(ν_sills)), mean(skipmissing(ν_nuggets))

# ╔═╡ cd7a3622-a4ee-46a5-bc7d-28c23be429d5
ν̂ = Matrix(Matrix(ν̂_df)')

# ╔═╡ f69f7433-5d6b-4fae-843e-7d47720332b1
md"Another role of residual kriging is to fill missing values that occur out of sampling duration for short-term monitoring sites."

# ╔═╡ b7203bec-bf89-4c36-9ccd-52796a553b6f
plot(heatmap(ν), heatmap(ν̂), size=(800,500), plot_title="ν field before vs. after kriging")

# ╔═╡ d97e7b6e-6f53-40a6-a406-93680985ceeb
md"### *f*: Estimate residual fields"

# ╔═╡ 2e26417c-7af8-42bb-8147-79e96ad6c58f
# For model
function krig_nu(c_itp_noextrap::Matrix, cov::DataFrame, μ::Matrix, range::Float64, sill::Float64, nugget::Float64)

	ν = c_itp_noextrap .- μ
	ν = replace(ν, NaN=>missing)
	ν_geo = georef(DataFrame(ν', :auto), PointSet(Matrix(cov[:, [:x, :y]])'))

	ν̂_df = DataFrame(zeros(size(c_itp_noextrap')), :auto)
	ν̂_var_df = DataFrame(zeros(size(c_itp_noextrap')), :auto)
	ν_vgram = ExponentialVariogram(range=range, sill=sill, nugget=nugget)
	
	for i in 1:size(c_itp_noextrap,1)
		ν_idx1 = ismissing.(ν[i,:]) .== false
		ν_geo1 = georef(DataFrame(ν=ν[i,:][ν_idx1]),
			PointSet((Matrix(cov[:, [:x, :y]])[ν_idx1,:])'))
		ν_problem1 = EstimationProblem(ν_geo1, ν_geo.geometry, :ν)
		ν_solver1 = KrigingSolver(:ν => (variogram=ν_vgram, mean=0.0))
		ν_sol1 = solve(ν_problem1, ν_solver1)
		ν̂_df[:,i] = ν_sol1.ν
		ν̂_var_df[:,i] = ν_sol1.ν_variance
	end
	
	return (ν̂=Matrix(Matrix(ν̂_df)'), ν_var=Matrix(Matrix(ν̂_var_df)'))
end

# ╔═╡ 8e8c0e33-2807-49a4-ac8e-76c02478bf74
# For estimation
function krig_nu(c_itp_noextrap::Matrix, cov::DataFrame, μ::Matrix, cov_new::DataFrame, range::Float64, sill::Float64, nugget::Float64)

	ν = c_itp_noextrap .- μ
	ν = replace(ν, NaN=>missing)
	ν_geo = georef(DataFrame(ν', :auto), PointSet(Matrix(cov[:, [:x, :y]])'))

	ν̂_df = DataFrame(zeros(size(cov_new,1),size(c_itp_noextrap,1)), :auto)
	ν̂_var_df = DataFrame(zeros(size(cov_new,1),size(c_itp_noextrap,1)), :auto)
	ν_vgram = ExponentialVariogram(range=range, sill=sill, nugget=nugget)
	
	for i in 1:size(c_itp_noextrap,1)
		ν_idx1 = ismissing.(ν[i,:]) .== false
		ν_geo1 = georef(DataFrame(ν=ν[i,:][ν_idx1]),
			PointSet((Matrix(cov[:, [:x, :y]])[ν_idx1,:])'))
		ν_problem1 = EstimationProblem(ν_geo1, 
			PointSet(Matrix(cov_new[:, [:x, :y]])'), :ν)
		ν_solver1 = KrigingSolver(:ν => (variogram=ν_vgram, mean=0.0))
		ν_sol1 = solve(ν_problem1, ν_solver1)
		ν̂_df[:,i] = ν_sol1.ν
		ν̂_var_df[:,i] = ν_sol1.ν_variance
	end
	
	return (ν̂=Matrix(Matrix(ν̂_df)'), ν_var=Matrix(Matrix(ν̂_var_df)'))
end

# ╔═╡ 520422ee-e19f-4e57-94ac-a684d99f7d74
md"""
## Overall predictions for the sites

Finally we can make predictions (`ŷ`) based on the time trends, the pls spatial model, and the residual model, and compare it to our original data.

"""

# ╔═╡ 765f6146-3572-4106-9385-1abddfa70474
ŷ = μ̂ .+ ν̂;

# ╔═╡ 6ccaf0a9-1192-44e3-8308-ef513e43c51f
let
	p_total = []
	for i in 1:nsite
		p1 = scatter(nh3_itp_noextrap[:,i], xticks=false, 
			label = "Obs_int", legend_position=:topright, 
			title=labels[i])
		plot!(μ̂[:,i], 
			ls=:dash, label="Mean")
		plot!(ŷ[:,i], 
			label="Total")
		push!(p_total, p1)
	end
	p = plot(p_total..., layouts=(:,2), size=(900,900), plot_title="Overall predictions")
end

# ╔═╡ aaf98e73-dc63-4b2a-a458-5578e46254a8
md"Here's our R2 score:"

# ╔═╡ 69551f66-ecc5-4410-bfc5-70f0aca98f27
Metrics.r2_score(nh3_itp_noextrap[keepidx], ŷ[keepidx])

# ╔═╡ c7be42d9-99d2-4ce5-9d30-27e0397f0c49
md"""
## Cross validation

Model performance is evaluated by leave-one-out cross-validation, that is, building a model with data of nine locations, and then making predictions (`y_test`) for the rest location using the model.
"""

# ╔═╡ 96736c70-fd33-4715-b301-96527bb4e08e
md"### *f*: Cross validation "

# ╔═╡ 68acd1c1-e241-4c05-967c-ef8439ce8aca
function cv_fin_mod(c_itp, c_itp_noexp, cov, tt, pls, alpha, beta_means, nums, beta_range, beta_sill, beta_nugget)
	c_itp1, c_itp_noexp1, cov1, cov_mat1 = subsample_data(c_itp, c_itp_noexp, cov, nums)
	
	beta1 = regress_beta(c_itp_noexp1, tt)

	pls_scores1 = pls[nums,:,:]

	alpha1 = regress_alpha(pls_scores1, beta1)
	
	beta_bar1 = calc_beta_means(alpha1, pls_scores1, beta_means)
	beta_hat1 = krig_beta(beta1, beta_bar1, cov1, beta_range, beta_sill, beta_nugget).β̂
	mu1 = calc_mu(tt, beta_hat1)

	# Only for testing function
	# nu1 = krig_nu(c_itp_noexp1, cov1, mu1, ν_range, ν_sill, ν_nugget).ν̂; mu1 .+ nu1 == ŷ
	
	return (c_itp_noexp=c_itp_noexp1, cov=cov1, beta=beta1, alpha=alpha1, beta_bar=beta_bar1, mu=mu1)
end

# ╔═╡ 7036e3e7-d973-4748-907b-e03a263c30a0
function cv_fin_test(c_itp, c_itp_noexp, cov, tt, pls, alpha, beta_means, test_num, beta_range, beta_sill, beta_nugget, nu_range, nu_sill, nu_nugget)
	# Modeling with modeling set
	mod_nums = Vector(1:size(c_itp,2))
	filter!(e -> e != test_num, mod_nums)  # Comment out this line for test
	model1 = cv_fin_mod(c_itp, c_itp_noexp, cov, tt, pls, alpha, beta_means, mod_nums, beta_range, beta_sill, beta_nugget)

	# Prediction of testing set
	cov1 = subsample_data(c_itp, c_itp_noexp, cov, [test_num,]).cov
	
	pls_scores1 = reshape(pls[test_num,:,:], 1, size(pls,2), size(pls,3))
	
	beta_bar1 = calc_beta_means(model1.alpha, pls_scores1, beta_means)
	beta_hat1 = krig_beta(model1.beta, model1.beta_bar, model1.cov, beta_bar1, cov1, beta_range, beta_sill, beta_nugget).β̂
	mu1 = calc_mu(tt, beta_hat1)
	nu1 = krig_nu(model1.c_itp_noexp, model1.cov, model1.mu, cov1, nu_range, nu_sill, nu_nugget).ν̂
	y1 = mu1 .+ nu1

	return (y=y1, mu=mu1, nu=nu1)
	# return (y=calc_mu(tt, beta_bar1), mu=mu1, nu=nu1)  # w/o kriging
end

# ╔═╡ 5e9edc9b-4435-4225-9b43-66a37ce5fda8
function cv_fin_stats(c_itp_noexp, cov, y; scale)
	
	nsite1 = size(y,2)
	rmse = zeros(nsite1)
	r2_cv = zeros(nsite1)
	r2_reg = zeros(nsite1)
	
	lta0 = mean(10 .^ timetrend_pred, dims=1)[:]
	lta1 = mean(10 .^ y, dims=1)[:]  # zeros(nsite1)

	present = isnan.(c_itp_noexp) .== false

	if scale in [:normal, :original]
		c_itp_noexp = 10 .^ c_itp_noexp
		y = 10 .^ y
	else
		lta0 = log10.(lta0)
		lta1 = log10.(lta1)
	end
	
	for i ∈ 1:nsite1
		y0 = c_itp_noexp[present[:,i],i]
		y1 = y[present[:,i],i]

		rmse[i] = sqrt(Metrics.mse(y1,y0))
		r2_cv[i] = max(0., Metrics.r2_score(y1,y0))
		r2_reg[i] = Statistics.cor(y1,y0)^2
		
		# lta0[i] = mean(y0)
		# lta1[i] = mean(y[:,i])  # mean(y1)
	end

	y0 = c_itp_noexp[present]
	y1 = y[present]
	
	df1 = DataFrame(ID=cov.ID, LTA_obs=lta0, LTA_CV=lta1,
		RMSE=rmse, R2_CV=r2_cv, R2_reg=r2_reg)
	df2 = DataFrame(Name=["Mean","LTA","Whole-site"], 
		RMSE=[mean(rmse), sqrt(Metrics.mse(lta1,lta0)), sqrt(Metrics.mse(y1,y0))],
		R2_CV=[mean(r2_cv), max(0., Metrics.r2_score(lta1,lta0)), 
			max(0., Metrics.r2_score(y1,y0))],
		R2_reg=[mean(r2_reg), Statistics.cor(lta1,lta0)^2, Statistics.cor(y1,y0)^2])
	
	return df1, df2
end

# ╔═╡ 83f7a833-8bf7-4c31-b4e5-a59f42392eeb
function cv_fin_avg(tt_pred, ds, cov, y)
	rmse = zeros(2)
	r2_cv = zeros(2)
	r2_reg = zeros(2)
	
	# Long-term average
	lta0 = mean(tt_pred, dims=1)[:]
	lta1 = mean(y, dims=1)[:]
	rmse[1] = sqrt(Metrics.mse(lta1,lta0))
	r2_cv[1] = max(0., Metrics.r2_score(lta1,lta0))
	r2_reg[1] = Statistics.cor(lta1,lta0)^2

	# Quarter average
	qa0 = zeros(size(tt_pred,2), 4)
	qa1 = deepcopy(qa0)
	Qs = Dates.quarterofyear.(Dates.unix2datetime.(ds))
	for Q in 1:4
		iq = Qs .== Q
		qa0[:,Q] = mean(tt_pred[iq,:], dims=1)
		qa1[:,Q] = mean(y[iq,:], dims=1)
	end

	rmse[2] = sqrt(Metrics.mse(qa1[:],qa0[:]))
	r2_cv[2] = max(0., Metrics.r2_score(qa1[:],qa0[:]))
	r2_reg[2] = Statistics.cor(qa1[:],qa0[:])^2
	
	avg = hcat(DataFrame(ID=cov.ID), 
		DataFrame(hcat(lta0,qa0,lta1,qa1), 
			[:LTA_obs,:Q1_obs,:Q2_obs,:Q3_obs,:Q4_obs,
				:LTA_CV,:Q1_CV,:Q2_CV,:Q3_CV,:Q4_CV]))
	fitness = DataFrame(Name=["LTA","QA"],
		RMSE=rmse, R2_CV=r2_cv, R2_reg=r2_reg)
	
	return avg, fitness
end

# ╔═╡ 1ea7940c-a4de-436c-acd1-743257a5b1bb
md"### Cross-validation results"

# ╔═╡ 5695bfb8-abe9-46c2-a22d-d1f7d5903388
begin
	y_test = zeros(size(nh3_itp))
	mu_test = zeros(size(nh3_itp))
	
	for i ∈ 1:nsite
		result1 = cv_fin_test(nh3_itp, nh3_itp_noextrap, geovars_ordered, 	
			smooth_timetrends, pls_scores, α, mean(β, dims=2), i, 
			β_range, β_sill, β_nugget, ν_range, ν_sill, ν_nugget)
		y_test[:,i] = result1.y
		mu_test[:,i] = result1.mu
	end
end

# ╔═╡ 10f47514-3dfc-4015-924c-2bffc8911ec9
md"""
Here, we use three measures of fitness:

1. **Root mean square error (RMSE)**
2. **Cross-validation R² (R2_CV)** represents the extent of fitness to the 1:1 line
3. **Linear regression R² (R2_reg)** represents the extent of fitness to the best linear model with intercept
"""

# ╔═╡ 735affae-37b6-4c0f-94fc-9198879b065c
cv_stats = cv_fin_stats(nh3_itp_noextrap, geovars_ordered, y_test; scale=:log)

# ╔═╡ 8f2bfa1b-1c39-4d13-b370-875723f01134
let
	p = scatter(nh3_itp_noextrap, y_test, label=labels)
	scatter!(cv_stats[1].LTA_obs, cv_stats[1].LTA_CV, 
		shape=:xcross, mc=:black, markerstrokewidth=4, label="LTA")
	Plots.abline!(1, 0, label="1:1", c=:black, ls=:dash, 
			xlab="Observation", ylab="Prediction",
			legend_position=:outerright)
end

# ╔═╡ b79020c5-058e-450a-b1f9-4c39020c69c1
let
	p_total = []
	for i in 1:nsite
		p1 = scatter(nh3_itp_noextrap[:,i], xticks=false, 
			#xticks=itp_time_ticks, x_rotation=90, 
			label = "Obs", legend_position=:topright, 
			title=labels[i])
		plot!(mu_test[:,i], label="Mean", ls=:dash)
		plot!(y_test[:,i], label="Overall")
		push!(p_total, p1)
	end
	p = plot(p_total..., layouts=(:,2), size=(900,900), plot_title="Time-series predictions in cross-validation")
	# savefig(p, "Figs/fin_cv_each_240617.pdf")
end

# ╔═╡ 4a775edb-6aa8-4d90-a29d-0d649a9553c4
md"""
## Fine-scale predictions
In this section, we use our model and geospatial data to make predictions for the entire area.
"""

# ╔═╡ be392d22-8695-4123-8d85-4e9855542780
md"""
### Data loading and screening

First, we need to load the spatial grid (`grid`) on which we will make predictions, and geographic covariates computed for the grid (`geovars_grid`).
"""

# ╔═╡ 8db4c4d6-aa41-406b-af18-cdb9ebbcaba0
grid = CSV.read("grid.csv", DataFrame)

# ╔═╡ 92d0de40-a6d9-40e7-9eb1-20571a132e81
begin
	xmin, ymin = minimum(Matrix(grid), dims=1)
	xmax, ymax = maximum(Matrix(grid), dims=1)
	xgrid=xmin/1000:0.03:xmax/1000
	ygrid=ymin/1000:0.03:ymax/1000
end

# ╔═╡ 1af77e8a-7a1a-41cd-9abc-f0a0aa0f2b26
begin
	geovars_grid = CSV.read("cov_roi.csv", DataFrame)
	geovars_grid1 = DataFrame(geovars_grid[:,:])
	rename!(geovars_grid1, geovars_names)
	geovars_grid_mat = let
		x_ref = Matrix(geovars_ordered1[ididx, 
			Not("ID", "lat", "long", "x", "y", "type")])
		x = Matrix(geovars_grid)
		skew = [skewness(x_ref[:,i]) for i in 1:size(x_ref,2)]
		for i in 1:size(x_ref,2)
			if skew[i] > skew_threshold
				x[:,i] = log10.(x[:,i] .+ .1)
			elseif skew[i] < -skew_threshold
				x[:,i] = -log10.(1000. .- x[:,i])
			end
		end
		x
	end
	geovars_grid[:,:] = geovars_grid_mat[:,:]
	geovars_grid = hcat(grid, geovars_grid)
end;

# ╔═╡ b12bdd7f-1f8f-4be3-a670-1a2bb9168504
rename!(geovars_grid, ["x", "y", geovars_names...])

# ╔═╡ 8be337c4-9337-4e6a-ba82-30683ed15bcb
md"### Predictions at selected time points"

# ╔═╡ 38714795-add5-4d68-9fbd-90f7b9698d5e
md"Let's set some time points (`t_idx`) to make predictions (as a matrix in `y_grid` and as a data frame in `y_grid_df`) at:"

# ╔═╡ f8a5578d-5c60-45a7-bc94-8f251f388bfe
begin
	t_idx = [5,12,18,25]
	Date.(Dates.unix2datetime.(biweekly))[t_idx]
end

# ╔═╡ b68663bb-c948-4609-af81-8c60e54bc0e8
β̂_grid = let
	cov_std1 = standardize_data(geovars_grid_mat, std_machine)
	pls_scores1 = calc_pls(cov_std1, pls_w, num_timetrends)
	beta_bar1 = calc_beta_means(α, pls_scores1, mean(β, dims=2))
	beta_hat1 = krig_beta(β, β̄, geovars_ordered, beta_bar1, geovars_grid, β_range, β_sill, β_nugget).β̂	
end

# ╔═╡ 920f88d9-24d3-4626-88a9-403f862e4b70
y_grid = let
	mu1 = calc_mu(smooth_timetrends[t_idx,:], β̂_grid)
	nu1 = krig_nu(nh3_itp_noextrap[t_idx,:], geovars_ordered, μ̂[t_idx,:], geovars_grid, ν_range, ν_sill, ν_nugget).ν̂
	mu1 .+ nu1
end

# ╔═╡ 3ead108a-f356-4b6b-ad3f-fd8610992175
DataFrame(date=Date.(Dates.unix2datetime.(biweekly))[t_idx],
	min=vec(10 .^ minimum(y_grid, dims=2)),
	p25=vec(10 .^ [percentile(y_grid[i,:], 25) for i in 1:length(t_idx)]),
	p50=vec(10 .^ median(y_grid, dims=2)),
	p75=vec(10 .^ [percentile(y_grid[i,:], 75) for i in 1:length(t_idx)]),
	max=vec(10 .^ maximum(y_grid, dims=2)))

# ╔═╡ 5dcfcd15-e818-4676-927d-786f7aef53f5
let
	p1 = [heatmap(xgrid, ygrid, reshape(β̂_grid[i,:],234,234)',
		title="β$(i-1)", cmap=:vik,
	) for i in 1:num_timetrends+1]
	p = plot(p1..., layout=(:,2), size=(950,800), plot_title="Predicted βs")
	
end

# ╔═╡ c3afb1dc-6b77-4e83-af5f-d17eea44bdc8
let
	p1 = [heatmap(xgrid, ygrid, reshape(y_grid[i,:],234,234)',
		title="$(Date.(Dates.unix2datetime.(biweekly))[t_idx[i]])",
		cmap=:vik,
	) for i in 1:length(t_idx)]
	p = plot(p1..., layout=(:,2), size=(950,800), plot_title="Predicted NH₃ surface (log-transformed)")
end

# ╔═╡ cf348dbb-4b85-483b-8fb9-82a34da082bb
y_grid_df = let
	df1 = DataFrame(y_grid', string.(Date.(Dates.unix2datetime.(biweekly))[t_idx]))
	hcat(grid, df1)
end

# ╔═╡ f77d7d1f-3279-4915-a6b7-2e861504dd81
md"""
### Predictions at specific locations

Our model can predict the temporal variation at specific location. Here, for example, we will make predictions at a longitude (or an x). Set the approximate x in meters to look at:
"""

# ╔═╡ f49c3d1a-1a89-4ae8-9b1f-7810ca394af1
@bind x NumberField(minimum(grid.x):maximum(grid.x), default=395960)

# ╔═╡ ca258b13-5d4e-4bbd-8f7a-50258d1be3b2
# Extract grid
begin
	line_idx0 = findfirst(grid.x .>= x)-1
	grid_line = grid[line_idx0:234:length(grid.x),:]  
	print("idx=", line_idx0, ", x=", grid_line.x[1])
end

# ╔═╡ 9103f75a-873c-4d66-bcc5-b5b71932c3c3
β̂_line = β̂_grid[:, line_idx0:234:length(grid.x)]

# ╔═╡ 2fc14dd3-8f3c-4176-a80d-5f4231f29ba8
y_line = let
	mu1 = calc_mu(smooth_timetrends, β̂_line)
	nu1 = krig_nu(nh3_itp_noextrap, geovars_ordered, μ̂, grid_line, ν_range, ν_sill, ν_nugget).ν̂
	mu1 .+ nu1
end

# ╔═╡ d1f75886-dc30-4020-af5e-4c3b0dd2149c
heatmap(Dates.unix2datetime.(biweekly), ygrid, y_line', c=:heat,
	title = "Predictions at a longitude (log-transformed)",
	ylabel = "Northing (m)",
	size = (800,300), left_margin = 4Plots.mm,
)

# ╔═╡ eb30a1fb-9621-45c0-9acf-92549374cd1e
plot(ygrid, mean(y_line, dims=1)[:], ribbon=std(y_line, dims=1)[:], label=false,
	c=1, fillalpha=0.2,
	title = "Temporal mean and STD at a fixed longitude",
	xlabel = "Northing (m)",
	ylabel = "NH₃ (μg/m³)",
	yticks=(log10.([.5,1.,2.,5.,10.,20.,50.]),[.5,1.,2.,5.,10.,20.,50.]),
	)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
GeoStats = "dcc97b0b-8ce5-5539-9008-bb190f959ef6"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
KissSmoothing = "23b0397c-cd08-4270-956a-157331f0528f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
MLJBase = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
MLJLinearModels = "6ee0df7b-362f-4a72-a706-9e79364fb692"
MLJModels = "d491faf4-2d78-11e9-2867-c94bc002c0b7"
Metrics = "cb9f3049-315b-4f05-b90c-a8adaec4da78"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
PartialLeastSquaresRegressor = "f4b1acfe-f311-436c-bb79-8483f53c17d5"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.10.15"
DataFrames = "~1.7.0"
GeoStats = "~0.47.4"
Interpolations = "~0.15.1"
KissSmoothing = "~1.0.8"
MLJ = "~0.20.0"
MLJBase = "~1.7.0"
MLJLinearModels = "~0.10.0"
MLJModels = "~0.16.12"
Metrics = "~0.1.2"
MultivariateStats = "~0.10.3"
PartialLeastSquaresRegressor = "~2.2.0"
Plots = "~1.40.13"
PlutoUI = "~0.7.62"
StatsBase = "~0.33.21"
StatsPlots = "~0.15.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "ab2992c1871de072662e9800cac3106accc03b87"

[[deps.ARFFFiles]]
deps = ["CategoricalArrays", "Dates", "Parsers", "Tables"]
git-tree-sha1 = "678eb18590a8bc6674363da4d5faa4ac09c40a18"
uuid = "da404889-ca92-49ff-9e8b-0aa6b4d38dc8"
version = "1.5.0"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "0ba8f4c1f06707985ffb4804fdad1bf97b233897"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.41"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cde29ddf7e5726c9fb511f340244ea3481267608"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "680b3b8759bd4c54052ada14e52355ab69e07876"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "c5aeb516a84459e0318a02507d2261edad97eb75"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.7.1"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "e2144b631226d9eeab2d746ca8880b7ccff504ae"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.3"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "2ac646d71d0d24b44f3f8c84da8c9f4d70fb67df"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.4+0"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.CategoricalDistributions]]
deps = ["CategoricalArrays", "Distributions", "Missings", "OrderedCollections", "Random", "ScientificTypes"]
git-tree-sha1 = "926862f549a82d6c3a7145bc7f1adff2a91a39f0"
uuid = "af321ab8-2d2e-40a6-b165-3d674595d28e"
version = "0.1.15"

    [deps.CategoricalDistributions.extensions]
    UnivariateFiniteDisplayExt = "UnicodePlots"

    [deps.CategoricalDistributions.weakdeps]
    UnicodePlots = "b8865327-cd53-5732-bb35-84acbb429228"

[[deps.Chain]]
git-tree-sha1 = "8c4920235f6c561e401dfe569beb8b924adad003"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.5.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CircularArrays]]
deps = ["OffsetArrays"]
git-tree-sha1 = "e24a6f390e5563583bb4315c73035b5b3f3e7ab4"
uuid = "7a955b69-7140-5f4e-a0ed-f168c5e2e749"
version = "1.4.0"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

[[deps.CoDa]]
deps = ["AxisArrays", "Distances", "Distributions", "FillArrays", "LinearAlgebra", "Printf", "Random", "StaticArrays", "Statistics", "Tables"]
git-tree-sha1 = "302e6be5786411dd09dc9df962ae60d54ac4b0bb"
uuid = "5900dafe-f573-5c72-b367-76665857777b"
version = "1.5.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b5278586822443594ff615963b0c09755771b3e0"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.26.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.ColumnSelectors]]
git-tree-sha1 = "221157488d6e5942ef8cc53086cad651b632ed4e"
uuid = "9cc86067-7e36-4c61-b350-1ac9833d277f"
version = "0.1.1"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataScienceTraits]]
git-tree-sha1 = "86a2d08f4b63f81abc16fdd9d2120cfe46e2261a"
uuid = "6cb2f572-2d2b-4ba6-bdb3-e710fa044d6c"
version = "0.1.0"

    [deps.DataScienceTraits.extensions]
    DataScienceTraitsCategoricalArraysExt = "CategoricalArrays"
    DataScienceTraitsCoDaExt = "CoDa"
    DataScienceTraitsDynamicQuantitiesExt = "DynamicQuantities"
    DataScienceTraitsUnitfulExt = "Unitful"

    [deps.DataScienceTraits.weakdeps]
    CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
    CoDa = "5900dafe-f573-5c72-b367-76665857777b"
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DecisionTree]]
deps = ["AbstractTrees", "DelimitedFiles", "LinearAlgebra", "Random", "ScikitLearnBase", "Statistics"]
git-tree-sha1 = "526ca14aaaf2d5a0e242f3a8a7966eb9065d7d78"
uuid = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
version = "0.12.4"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DensityRatioEstimation]]
deps = ["ChainRulesCore", "GPUArraysCore", "LinearAlgebra", "Parameters", "Random", "Statistics", "StatsBase"]
git-tree-sha1 = "22a98b13abd7b2862fdfb205c53d09b4b8d9d807"
uuid = "ab46fb84-d57c-11e9-2f65-6f72e4a7229f"
version = "1.3.1"

    [deps.DensityRatioEstimation.extensions]
    DensityRatioEstimationConvexExt = ["Convex", "ECOS"]
    DensityRatioEstimationJuMPExt = ["JuMP", "Ipopt"]
    DensityRatioEstimationOptimExt = "Optim"

    [deps.DensityRatioEstimation.weakdeps]
    Convex = "f65535da-76fb-5f13-bab9-19810c17039a"
    ECOS = "e2685f51-7e38-5353-a97d-a921fd2c8199"
    Ipopt = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
    JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
    Optim = "429524aa-4258-5aef-a3af-852621145aeb"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "a86af9c4c4f33e16a2b2ff43c2113b2f390081fa"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.4.5"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "0b4190661e8a4e51a842070e7dd4fae440ddb7f4"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.118"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "e7b7e6f178525d17c720ab9c081e4ef04429f860"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.4"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarlyStopping]]
deps = ["Dates", "Statistics"]
git-tree-sha1 = "98fdf08b707aaf69f524a6cd0a67858cefe0cfb6"
uuid = "792122b4-ca99-40de-a6bc-6742525f08b6"
version = "0.3.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "7de7c78d681078f027389e067864a8d53bd7c3c9"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "73d1214fec245096717847c62d389a5d2ac86504"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.22.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "301b5d5d731a0654825f1f2e906990f7141a106b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.16.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "273bd1cd30768a2fddfa3fd63bbc746ed7249e5f"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.9.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "7ffa4049937aeba2e5e1242274dc052b0362157a"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.14"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "98fc192b4e4b938775ecd276ce88f539bcec358e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.14+0"

[[deps.GeoStats]]
deps = ["CategoricalArrays", "Chain", "CoDa", "DataScienceTraits", "Dates", "DensityRatioEstimation", "Distances", "GeoStatsBase", "GeoStatsModels", "GeoStatsProcesses", "GeoStatsSolvers", "GeoStatsTransforms", "GeoTables", "LossFunctions", "Meshes", "Reexport", "Rotations", "Statistics", "StatsLearnModels", "TableTransforms", "Tables", "Unitful", "Variography"]
git-tree-sha1 = "d20ff0eef5dc3cc78e731666f0aad9fefeb3f417"
uuid = "dcc97b0b-8ce5-5539-9008-bb190f959ef6"
version = "0.47.4"

    [deps.GeoStats.extensions]
    GeoStatsMakieExt = "Makie"

    [deps.GeoStats.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"

[[deps.GeoStatsBase]]
deps = ["CategoricalArrays", "ColumnSelectors", "DataScienceTraits", "DensityRatioEstimation", "Distances", "Distributed", "Distributions", "GeoTables", "LinearAlgebra", "LossFunctions", "MLJModelInterface", "Meshes", "Optim", "ProgressMeter", "Random", "Rotations", "Statistics", "StatsBase", "TableTransforms", "Tables", "Transducers", "TypedTables", "Unitful"]
git-tree-sha1 = "1e2e9e4d4126ef6c56091270be28dd1e9374e87f"
uuid = "323cb8eb-fbf6-51c0-afd0-f8fba70507b2"
version = "0.40.0"

[[deps.GeoStatsModels]]
deps = ["Combinatorics", "Distances", "Distributions", "GeoTables", "LinearAlgebra", "Meshes", "Statistics", "Tables", "Unitful", "Variography"]
git-tree-sha1 = "43e7706a24b08973ff2d21a94c6403a6e1457270"
uuid = "ad987403-13c5-47b5-afee-0a48f6ac4f12"
version = "0.2.1"

[[deps.GeoStatsProcesses]]
deps = ["Bessels", "CpuId", "Distances", "Distributed", "Distributions", "FFTW", "GeoStatsBase", "GeoStatsModels", "GeoTables", "LinearAlgebra", "Meshes", "ProgressMeter", "Random", "Statistics", "Tables", "Variography"]
git-tree-sha1 = "c992da2bc67857188efb02d3004501619812085c"
uuid = "aa102bde-5a27-4b0c-b2c1-e7a7dcc4c3e7"
version = "0.3.4"

    [deps.GeoStatsProcesses.extensions]
    GeoStatsProcessesImageQuiltingExt = "ImageQuilting"
    GeoStatsProcessesStratiGraphicsExt = "StratiGraphics"
    GeoStatsProcessesTuringPatternsExt = "TuringPatterns"

    [deps.GeoStatsProcesses.weakdeps]
    ImageQuilting = "e8712464-036d-575c-85ac-952ae31322ab"
    StratiGraphics = "135379e1-83be-5ae7-9e8e-29dade3dc6c7"
    TuringPatterns = "fde5428d-3bf0-5ade-b94a-d334303c4d77"

[[deps.GeoStatsSolvers]]
deps = ["Bessels", "CpuId", "Distances", "Distributions", "FFTW", "GeoStatsBase", "GeoStatsModels", "GeoTables", "LinearAlgebra", "Meshes", "Random", "Statistics", "StatsLearnModels", "TableTransforms", "Tables", "Unitful", "Variography"]
git-tree-sha1 = "0102b6b5f25d33273507478bcd69ddf6b48bfe02"
uuid = "50e95529-e670-4fa6-84ad-e28f686cc091"
version = "0.7.10"

[[deps.GeoStatsTransforms]]
deps = ["ArnoldiMethod", "CategoricalArrays", "Clustering", "ColumnSelectors", "Combinatorics", "DataScienceTraits", "Distances", "GeoStatsModels", "GeoStatsProcesses", "GeoTables", "LinearAlgebra", "Meshes", "Random", "SparseArrays", "Statistics", "TableDistances", "TableTransforms", "Tables", "Unitful"]
git-tree-sha1 = "5635cfa0268bb34a20ed7d3eb3fc8766b58a260d"
uuid = "725d9659-360f-4996-9c94-5f19c7e4a8a6"
version = "0.2.3"

[[deps.GeoTables]]
deps = ["ColumnSelectors", "DataAPI", "DataScienceTraits", "Meshes", "PrettyTables", "Random", "Statistics", "Tables", "Unitful"]
git-tree-sha1 = "4d2cb731ceb24cc57c38b18020c1f77628a60492"
uuid = "e502b557-6362-48c1-8219-d30d308dcdb0"
version = "1.9.1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "f93655dc73d7a0b4a368e3c0bce296ae035ad76e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.16"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "6a9fde685a7ac1eb3495f8e812c5a7c3711c2d5e"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.3"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "0f14a5456bdc6b9731a5682f439a672750a09e48"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.0.4+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IterationControl]]
deps = ["EarlyStopping", "InteractiveUtils"]
git-tree-sha1 = "e663925ebc3d93c1150a7570d114f9ea2f664726"
uuid = "b3c1a2ee-3fec-4384-bf48-272ea71de57c"
version = "0.5.4"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "59545b0a2b27208b0650df0a46b8e3019f85055b"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "Requires", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "ed7167240f40e62d97c1f5f7735dea6de3cc5c49"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.18"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.KissSmoothing]]
deps = ["FFTW", "LinearAlgebra", "SparseArrays", "Statistics"]
git-tree-sha1 = "6d4c314981c5f1df153c23b69dfa295c5286e247"
uuid = "23b0397c-cd08-4270-956a-157331f0528f"
version = "1.0.8"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "839c82932db86740ae729779e610f07a1640be9a"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.6.3"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "88b916503aac4fb7f701bb625cd84ca5dd1677bc"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.29+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cd10d2cc78d34c0e2a3a36420ab607b611debfbb"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.7"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LatinHypercubeSampling]]
deps = ["Random", "StableRNGs", "StatsBase", "Test"]
git-tree-sha1 = "825289d43c753c7f1bf9bed334c253e9913997f8"
uuid = "a5e1c1ea-c99a-51d3-a14d-a9a37257b02d"
version = "1.9.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LearnAPI]]
deps = ["InteractiveUtils", "Statistics"]
git-tree-sha1 = "ec695822c1faaaa64cee32d0b21505e1977b4809"
uuid = "92ad9a40-7767-427a-9ee6-6e577f1266cb"
version = "0.1.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a31572773ac1b745e0343fe5e2c8ddda7a37e997"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "321ccef73a96ba828cd51f2ab5b9f917fa73945a"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "7f6be2e4cdaaf558623d93113d6ddade7b916209"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.11.4"
weakdeps = ["ChainRulesCore", "SparseArrays", "Statistics"]

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"
    LinearMapsSparseArraysExt = "SparseArrays"
    LinearMapsStatisticsExt = "Statistics"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.LossFunctions]]
deps = ["Markdown", "Requires", "Statistics"]
git-tree-sha1 = "8073538845227d3acf8a0c149bf0e150e60d0ced"
uuid = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
version = "0.11.2"
weakdeps = ["CategoricalArrays"]

    [deps.LossFunctions.extensions]
    LossFunctionsCategoricalArraysExt = "CategoricalArrays"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "5de60bc6cb3899cd318d80d627560fae2e2d99ae"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.0.1+1"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

[[deps.MLFlowClient]]
deps = ["Dates", "FilePathsBase", "HTTP", "JSON", "ShowCases", "URIs", "UUIDs"]
git-tree-sha1 = "32cee10a6527476bef0c6484ff4c60c2cead5d3e"
uuid = "64a0f543-368b-4a9a-827a-e71edb2a0b83"
version = "0.4.4"

[[deps.MLJ]]
deps = ["CategoricalArrays", "ComputationalResources", "Distributed", "Distributions", "LinearAlgebra", "MLJBase", "MLJEnsembles", "MLJFlow", "MLJIteration", "MLJModels", "MLJTuning", "OpenML", "Pkg", "ProgressMeter", "Random", "Reexport", "ScientificTypes", "StatisticalMeasures", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "58d17a367ee211ade6e53f83a9cc5adf9d26f833"
uuid = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
version = "0.20.0"

[[deps.MLJBase]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Dates", "DelimitedFiles", "Distributed", "Distributions", "InteractiveUtils", "InvertedIndices", "LearnAPI", "LinearAlgebra", "MLJModelInterface", "Missings", "OrderedCollections", "Parameters", "PrettyTables", "ProgressMeter", "Random", "RecipesBase", "Reexport", "ScientificTypes", "Serialization", "StatisticalMeasuresBase", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "6f45e12073bc2f2e73ed0473391db38c31e879c9"
uuid = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
version = "1.7.0"
weakdeps = ["StatisticalMeasures"]

    [deps.MLJBase.extensions]
    DefaultMeasuresExt = "StatisticalMeasures"

[[deps.MLJEnsembles]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Distributed", "Distributions", "MLJModelInterface", "ProgressMeter", "Random", "ScientificTypesBase", "StatisticalMeasuresBase", "StatsBase"]
git-tree-sha1 = "84a5be55a364bb6b6dc7780bbd64317ebdd3ad1e"
uuid = "50ed68f4-41fd-4504-931a-ed422449fee0"
version = "0.4.3"

[[deps.MLJFlow]]
deps = ["MLFlowClient", "MLJBase", "MLJModelInterface"]
git-tree-sha1 = "dc0de70a794c6d4c1aa4bde8196770c6b6e6b550"
uuid = "7b7b8358-b45c-48ea-a8ef-7ca328ad328f"
version = "0.2.0"

[[deps.MLJIteration]]
deps = ["IterationControl", "MLJBase", "Random", "Serialization"]
git-tree-sha1 = "ad16cfd261e28204847f509d1221a581286448ae"
uuid = "614be32b-d00c-4edb-bd02-1eb411ab5e55"
version = "0.6.3"

[[deps.MLJLinearModels]]
deps = ["DocStringExtensions", "IterativeSolvers", "LinearAlgebra", "LinearMaps", "MLJModelInterface", "Optim", "Parameters"]
git-tree-sha1 = "7f517fd840ca433a8fae673edb31678ff55d969c"
uuid = "6ee0df7b-362f-4a72-a706-9e79364fb692"
version = "0.10.0"

[[deps.MLJModelInterface]]
deps = ["REPL", "Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "66626f80d5807921045d539b4f7153b1d47c5f8a"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.11.1"

[[deps.MLJModels]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Combinatorics", "Dates", "Distances", "Distributions", "InteractiveUtils", "LinearAlgebra", "MLJModelInterface", "Markdown", "OrderedCollections", "Parameters", "Pkg", "PrettyPrinting", "REPL", "Random", "RelocatableFolders", "ScientificTypes", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "10d221910fc3f3eedad567178ddbca3cc0f776a3"
uuid = "d491faf4-2d78-11e9-2867-c94bc002c0b7"
version = "0.16.12"

[[deps.MLJTuning]]
deps = ["ComputationalResources", "Distributed", "Distributions", "LatinHypercubeSampling", "MLJBase", "ProgressMeter", "Random", "RecipesBase", "StatisticalMeasuresBase"]
git-tree-sha1 = "38aab60b1274ce7d6da784808e3be69e585dbbf6"
uuid = "03970b2e-30c4-11ea-3135-d1576263f10f"
version = "0.8.8"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "MLCore", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "a772d8d1987433538a5c226f79393324b55f7846"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.8"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Meshes]]
deps = ["Bessels", "CircularArrays", "Distances", "LinearAlgebra", "NearestNeighbors", "Random", "Rotations", "SparseArrays", "StaticArrays", "StatsBase", "TransformsBase", "Unitful"]
git-tree-sha1 = "446aabed4ea4dfcaf72733f950ffdcda223346a9"
uuid = "eacbb407-ea5a-433e-ab97-5258b1ca43fa"
version = "0.35.21"

    [deps.Meshes.extensions]
    MeshesMakieExt = "Makie"

    [deps.Meshes.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"

[[deps.Metrics]]
deps = ["DataFrames", "DataStructures", "Random", "StatsBase"]
git-tree-sha1 = "6e9e77751dd230b360c29e23a10f6e6d2f4fafaf"
uuid = "cb9f3049-315b-4f05-b90c-a8adaec4da78"
version = "0.1.2"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "4abc63cdd8dd9dd925d8e879cda280bedc8013ca"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.30"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "030ea22804ef91648f29b7ad3fc15fa49d0e6e71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.3"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "8a3271d8309285f4db73b4f662b1b290c715e85e"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.21"

[[deps.NelderMead]]
git-tree-sha1 = "25abc2f9b1c752e69229f37909461befa7c1f85d"
uuid = "2f6b4ddb-b4ff-44c0-b59b-2ab99302f970"
version = "0.4.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenML]]
deps = ["ARFFFiles", "HTTP", "JSON", "Markdown", "Pkg", "Scratch"]
git-tree-sha1 = "63603b2b367107e87dbceda4e33c67aed17e50e0"
uuid = "8b6db2d4-7670-4922-a472-f9537c81ab66"
version = "0.3.2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9216a80ff3682833ac4b733caa8c00390620ba5d"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.0+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "c1f51f704f689f87f28b33836fd460ecf9b34583"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.11.0"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3b31172c032a1def20c98dae3f2cdc9d10e3b561"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.1+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "44f6c1f38f77cafef9450ff93946c53bd9ca16ff"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.2"

[[deps.PartialLeastSquaresRegressor]]
deps = ["LinearAlgebra", "MLJModelInterface", "Random", "Statistics"]
git-tree-sha1 = "21e053e5d9aed2ab883a5365fdcbf28729d325cf"
uuid = "f4b1acfe-f311-436c-bb79-8483f53c17d5"
version = "2.2.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "809ba625a00c605f8d00cd2a9ae19ce34fc24d68"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.13"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "d3de2694b52a01ce61a036f18ea9c0f61c4a9230"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.62"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyPrinting]]
git-tree-sha1 = "142ee93724a9c5d04d78df7006670a93ed1b244e"
uuid = "54e16d92-306c-5ea0-a30b-337be88ac337"
version = "0.4.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "66b20dd35966a748321d3b2537c4584cf40387c7"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "13c5103482a8ed1536a54c08d0e742ae3dca2d42"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.4"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "994cc27cdacca10e68feb291673ec3a76aa2fae9"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.6"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "5680a9276685d392c87407df00d57c9924d9f11e"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.7.1"
weakdeps = ["RecipesBase"]

    [deps.Rotations.extensions]
    RotationsRecipesBaseExt = "RecipesBase"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "75ccd10ca65b939dab03b812994e571bf1e3e1da"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "3.0.2"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "1147f140b4c8ddab224c94efa9569fc23d63ab44"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.3.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplitApplyCombine]]
deps = ["Dictionaries", "Indexing"]
git-tree-sha1 = "c06d695d51cfb2187e6848e98d6252df9101c588"
uuid = "03a91e81-4c3e-53e1-a0a4-9c0c8f19dd66"
version = "1.2.3"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0feb6b9031bd5c51f9072393eb5ab3efd31bf9e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.13"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.StatisticalMeasures]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Distributions", "LearnAPI", "LinearAlgebra", "MacroTools", "OrderedCollections", "PrecompileTools", "ScientificTypesBase", "StatisticalMeasuresBase", "Statistics", "StatsBase"]
git-tree-sha1 = "c1d4318fa41056b839dfbb3ee841f011fa6e8518"
uuid = "a19d573c-0a75-4610-95b3-7071388c7541"
version = "0.1.7"
weakdeps = ["LossFunctions", "ScientificTypes"]

    [deps.StatisticalMeasures.extensions]
    LossFunctionsExt = "LossFunctions"
    ScientificTypesExt = "ScientificTypes"

[[deps.StatisticalMeasuresBase]]
deps = ["CategoricalArrays", "InteractiveUtils", "MLUtils", "MacroTools", "OrderedCollections", "PrecompileTools", "ScientificTypesBase", "Statistics"]
git-tree-sha1 = "e4f508cf3b3253f3eb357274fe36fb3332ca9896"
uuid = "c062fc1d-0d66-479b-b6ac-8b44719de4cc"
version = "0.1.2"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "542d979f6e756f13f862aa00b224f04f9e445f11"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "35b09e80be285516e52c9054792c884b9216ae3c"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.4.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsLearnModels]]
deps = ["ColumnSelectors", "DecisionTree", "Distributions", "GLM", "TableTransforms", "Tables"]
git-tree-sha1 = "4091438ebf69623a6a23653eaaa62d455da8c048"
uuid = "c146b59d-1589-421c-8e09-a22e554fd05c"
version = "0.2.1"
weakdeps = ["MLJModelInterface"]

    [deps.StatsLearnModels.extensions]
    StatsLearnModelsMLJModelInterfaceExt = "MLJModelInterface"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "9022bcaa2fc1d484f1326eaa4db8db543ca8c66d"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.4"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

[[deps.StringDistances]]
deps = ["Distances", "StatsAPI"]
git-tree-sha1 = "5b2ca70b099f91e54d98064d5caf5cc9b541ad06"
uuid = "88034a9c-02f8-509d-84a9-84ec65e18404"
version = "0.11.3"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableDistances]]
deps = ["Distances", "Statistics", "StringDistances", "Tables"]
git-tree-sha1 = "6defeec6c838b305fde9778ec2417e9f3171bbe1"
uuid = "e5d66e97-8c70-46bb-8b66-04a2d73ad782"
version = "0.3.0"
weakdeps = ["CategoricalArrays", "CoDa"]

    [deps.TableDistances.extensions]
    TableDistancesCategoricalArraysExt = "CategoricalArrays"
    TableDistancesCoDaExt = "CoDa"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.TableTransforms]]
deps = ["AbstractTrees", "CategoricalArrays", "CoDa", "ColumnSelectors", "DataScienceTraits", "Distributions", "InverseFunctions", "LinearAlgebra", "NelderMead", "PrettyTables", "Random", "Statistics", "StatsBase", "Tables", "Transducers", "TransformsBase", "Unitful"]
git-tree-sha1 = "1bf737b0dbe298182ea3a4acf4dbb9bb92603555"
uuid = "0d432bfd-3ee1-4ac1-886a-39f05cc69a3e"
version = "1.21.0"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "5215a069867476fc8e3469602006b9670e68da23"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.82"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.TransformsBase]]
deps = ["AbstractTrees"]
git-tree-sha1 = "c209ddc5ea678a542aabe482713b24c9280919ed"
uuid = "28dd2a49-a57a-4bfb-84ca-1a49db9b96b8"
version = "1.6.0"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.TypedTables]]
deps = ["Adapt", "Dictionaries", "Indexing", "SplitApplyCombine", "Tables", "Unicode"]
git-tree-sha1 = "84fd7dadde577e01eb4323b7e7b9cb51c62c60d4"
uuid = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"
version = "1.4.6"

[[deps.URIs]]
git-tree-sha1 = "cbbebadbcc76c5ca1cc4b4f3b0614b3e603b5000"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "bf2c553f25e954a9b38c9c0593a59bb13113f9e5"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.5"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Variography]]
deps = ["Bessels", "Distances", "GeoTables", "InteractiveUtils", "LinearAlgebra", "Meshes", "NearestNeighbors", "Optim", "Printf", "Random", "Setfield", "Statistics", "StatsAPI", "Tables", "Transducers", "Unitful"]
git-tree-sha1 = "f94588756971a1f8a7cb6dd4ce224b3ba55346df"
uuid = "04a0146e-e6df-5636-8d7f-62fa9eb0b20c"
version = "0.19.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "e9aeb174f95385de31e70bd15fa066a505ea82b9"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.7"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b8b243e47228b4a3877f1dd6aee0c5d56db7fcf4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "9caba99d38404b285db8801d5c45ef4f4f425a6d"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.1+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0ba42241cb6809f1a278d0bcb976e0483c3f1f2d"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+1"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "068dfe202b0a05b8332f1e8e6b4080684b9c7700"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.47+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d5a767a3bb77135a99e433afe0eb14cd7f6914c3"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╠═551d6c28-78d3-47a5-8d86-9f0a10dd74b1
# ╠═02d73899-d3e6-4000-8a30-d3f967ab56e8
# ╠═d1b3ab98-5974-11ee-36dc-df38e5809536
# ╠═8806e90a-b911-4564-9d38-ff399ce7609f
# ╠═86a18d13-6c4d-4d23-8470-790ab4866d3a
# ╟─097921c2-eea4-4f24-a7a0-6473051b54f0
# ╠═fefc7d86-e2f6-44da-8128-0101cf1537ca
# ╟─3e899e68-956c-4e62-9e16-4e826c5a56c5
# ╠═bb41d9f2-c41a-4f9d-af10-3a849a50cea3
# ╟─740d509f-3fd3-44db-b1b3-2d81ecc40219
# ╠═6a1588fd-a8e5-49a7-b224-fb625161585f
# ╟─682f49e4-f345-443c-9077-21036029cbeb
# ╠═5c547e00-7fd1-44d1-b3aa-b22e89293d54
# ╠═41bb5af2-131f-4f6d-ba55-cc9e23c22006
# ╟─f6bdea65-f1cf-4c1c-9744-f75c8b9eedc0
# ╟─ac6ce0d4-d91c-46c2-82fb-dcc88a486b85
# ╟─df9c40bf-7e3d-480f-8ad2-7453b726068e
# ╠═c165f4fe-9422-46c7-bbd5-14786a790fb2
# ╟─b45ad166-8690-49fe-a29a-af55773d25e5
# ╟─13e650f3-1186-442e-84cc-33900e42936a
# ╟─09d0e599-56f1-4ef2-992e-92aa2df91022
# ╠═5fb05902-071e-4fee-9a35-021d16f20120
# ╠═2ce59e52-a7b5-4901-8bcc-fd3b3484be32
# ╟─d32d0503-66b5-42db-9609-e2fd181d1132
# ╠═eb1e84e5-81f8-4fff-ac75-a05111e81e26
# ╠═96d7d71a-c121-4066-9f7c-3815459461f0
# ╟─54d5d99f-4097-42e7-95bb-fc7193451ccf
# ╟─bdafc38a-dbdf-46a8-95eb-cc72fa937e3a
# ╟─eaa8ab2b-dc56-47ab-9eab-622a83c54f5e
# ╟─88f880dc-953d-4d46-a9f7-a315cff5704a
# ╟─4ed1da32-3cfb-4ef3-9cd9-87cb0f0e2f66
# ╠═b37e0dc1-2481-41d2-a923-170af872090d
# ╠═94f41856-fc33-4615-83f7-558888f13f14
# ╠═6fe4f6d3-ec1b-4b2a-8bea-6b881388382a
# ╟─4c9e6812-a3fa-4815-ab0e-96ef54b64fe5
# ╟─bf49745e-3fc1-4a40-9a24-a0495b2c8ec3
# ╟─3e3380bf-9a34-4538-b177-d661dedbbb35
# ╠═8ae9860f-0e38-4841-a0be-2f46f050844d
# ╠═478ee350-1092-475e-acd9-96f49faa2a6a
# ╟─d7458523-5253-4526-9b18-ea50373f2f79
# ╠═53d1ce27-4189-4d4b-9ec7-247530a88a3f
# ╟─bbcef19e-edbd-4f78-ba89-d0128c99922d
# ╟─49176761-5498-408a-831f-b6832651d517
# ╟─648a7ba6-be2c-4e98-8b54-41f13380448d
# ╠═306231bd-f42f-4a03-96dc-c994e1279084
# ╠═0971cb1e-b038-4c95-bf1a-98d84b3c04c9
# ╟─dc5b4a8c-3035-415c-ab69-60c987ab5200
# ╟─d96fcb89-3193-4bd5-9cf2-f0d90bf70e89
# ╟─5795c94d-ed24-40b5-9bf2-bdda3adfe7a4
# ╟─43da7ff6-34c3-4aa3-bcf9-ca1536ba22a5
# ╠═1200206c-a7af-423c-b655-6fb7c2e8e0b6
# ╟─8bdb179f-a565-429c-91fa-f3222286563d
# ╟─66b67223-02fc-4cd3-836b-fbd36e9dfab4
# ╟─907e5cdb-a332-4b77-9538-b10374c0d3e6
# ╠═a46e8775-f17a-4fe2-a3a6-85d50f80dcf9
# ╠═0224a5a1-839f-4a15-9401-e029a10004d7
# ╠═fa24dcb0-deaf-4cbd-bfe0-7e08bb3152c5
# ╠═b47aff02-2b0c-4861-8551-80773adace56
# ╟─1307d582-e272-4aea-bca7-101dc8797d6a
# ╟─ac379534-07e2-40d5-b4f8-cd9582113d0c
# ╟─91681a1e-c0ee-4dfc-b6fa-d5847ee3db22
# ╠═69c2b05a-43b5-4db0-902a-0b64121620f5
# ╟─b8768516-94b8-4727-bc9c-93da5ae9fc48
# ╟─b2a0739d-e21c-4a9b-a285-fb38e4c076fe
# ╠═04650a44-3062-4610-9638-2b36f3404c33
# ╟─be91c45c-6e58-4fc0-b05d-bfd40d68b30a
# ╟─34bac7af-961e-446b-84fa-11c2fd955b28
# ╠═4d92f4aa-dabf-4612-8f7e-ee94b4a4ff42
# ╟─27398978-0a4f-4a5e-b6a8-59c6baa1bbd5
# ╠═23f9a95b-e931-44b6-8c00-81f348345f27
# ╠═6a106804-0cea-492c-b502-7fcfba09bbb9
# ╟─11280080-e031-44da-a53b-ff2756359fe9
# ╟─4029a2ad-5a7a-4632-a1b9-28f6c71c6ecc
# ╟─fa0c624a-d3fe-4a6a-83ed-f9a72f81dc4d
# ╟─286067f0-61f9-4781-8217-39dc9f994895
# ╠═4c87672f-8038-48c2-90bd-0a53ef7a40eb
# ╠═e3e5dcce-78cf-4b77-b864-231742c5ef72
# ╠═fd8c02d7-1c37-4410-a26e-596abe95130f
# ╠═73103277-3434-44c7-b513-1ac4ef5beb50
# ╟─5e69fe99-1ea4-4aa8-847c-8e1e273639d5
# ╠═37213b75-8ca2-40a5-b8f5-056a86d08a2c
# ╟─5d7e353b-e9f2-4378-a970-846fb9a65c9e
# ╟─a2f0a3e9-b9e3-41d5-bf88-84c0eaae3dd0
# ╟─7f24b41a-9e8d-4313-81b0-552e3ebc13d0
# ╟─76997d3a-981f-4691-8c44-06e4956d2ee7
# ╟─f1c123b5-6c59-414e-aff1-66aeb660e365
# ╟─fb06a012-29c0-4dfa-bfca-6f8d8b820efc
# ╠═b728bf43-6853-417d-a705-f24849994b60
# ╠═424c7e26-83c2-4042-9529-619109d51c8d
# ╟─94e3091f-7ccc-4878-acbe-b7e93737b838
# ╠═6c1eec73-3e56-4a99-aca5-67b764be4f70
# ╠═ddce6b65-4439-4420-9bea-fe515fc57dde
# ╟─3fd55819-5b7f-4cb7-a459-83e700ffa5b0
# ╟─fdb7d310-51df-4133-85bd-92eb410122ed
# ╠═480344fd-e02a-4dc1-814f-a113c5743c0e
# ╠═88abe548-b780-488b-96c1-4201a50d5808
# ╠═7bced889-0373-4ad1-b433-cc8bd1ff0a00
# ╠═9d5313c0-9bfa-4b9f-8020-3b2b9df750ab
# ╠═21ed5ee0-6608-4153-9d58-06e2bcd57a76
# ╠═0b7d3b14-cc64-401f-908a-f426f9f517fc
# ╟─b49b10a8-b9da-4f0a-be9c-e29fb34b719c
# ╠═4ea7ab16-e830-4c6d-bda8-4e9f99f353ae
# ╟─54dca345-1615-465b-92b3-f268eb590675
# ╟─2715625a-a5d5-47af-b454-24c9fe929a31
# ╟─bd82c75e-096c-415f-9e6a-62e8b7a3b015
# ╟─237ab62e-c3c4-49ec-91b5-39fb7fbb08bc
# ╟─acd91af5-b39c-47d4-a02d-594b611ec51f
# ╠═57a91282-05ba-4165-a2e0-a7461e836610
# ╠═a0dd3b4c-335a-4d24-a28d-1f00dec7e4db
# ╠═da26eacf-11af-4e2e-be8d-9e8e5730afad
# ╟─db622d9c-9508-4c28-9265-8182a6f1c246
# ╠═aaec4bc0-27e9-4f34-bdfc-b4cdcfe61353
# ╠═c3c4cd2e-d03e-4294-86b2-3f9960777673
# ╠═cd7a3622-a4ee-46a5-bc7d-28c23be429d5
# ╟─f69f7433-5d6b-4fae-843e-7d47720332b1
# ╟─b7203bec-bf89-4c36-9ccd-52796a553b6f
# ╟─d97e7b6e-6f53-40a6-a406-93680985ceeb
# ╠═2e26417c-7af8-42bb-8147-79e96ad6c58f
# ╠═8e8c0e33-2807-49a4-ac8e-76c02478bf74
# ╟─520422ee-e19f-4e57-94ac-a684d99f7d74
# ╠═765f6146-3572-4106-9385-1abddfa70474
# ╟─6ccaf0a9-1192-44e3-8308-ef513e43c51f
# ╟─aaf98e73-dc63-4b2a-a458-5578e46254a8
# ╠═69551f66-ecc5-4410-bfc5-70f0aca98f27
# ╟─c7be42d9-99d2-4ce5-9d30-27e0397f0c49
# ╟─96736c70-fd33-4715-b301-96527bb4e08e
# ╠═68acd1c1-e241-4c05-967c-ef8439ce8aca
# ╠═7036e3e7-d973-4748-907b-e03a263c30a0
# ╠═5e9edc9b-4435-4225-9b43-66a37ce5fda8
# ╠═83f7a833-8bf7-4c31-b4e5-a59f42392eeb
# ╟─1ea7940c-a4de-436c-acd1-743257a5b1bb
# ╠═5695bfb8-abe9-46c2-a22d-d1f7d5903388
# ╟─10f47514-3dfc-4015-924c-2bffc8911ec9
# ╠═735affae-37b6-4c0f-94fc-9198879b065c
# ╟─8f2bfa1b-1c39-4d13-b370-875723f01134
# ╟─b79020c5-058e-450a-b1f9-4c39020c69c1
# ╟─4a775edb-6aa8-4d90-a29d-0d649a9553c4
# ╟─be392d22-8695-4123-8d85-4e9855542780
# ╠═8db4c4d6-aa41-406b-af18-cdb9ebbcaba0
# ╠═92d0de40-a6d9-40e7-9eb1-20571a132e81
# ╠═1af77e8a-7a1a-41cd-9abc-f0a0aa0f2b26
# ╠═b12bdd7f-1f8f-4be3-a670-1a2bb9168504
# ╟─8be337c4-9337-4e6a-ba82-30683ed15bcb
# ╟─38714795-add5-4d68-9fbd-90f7b9698d5e
# ╠═f8a5578d-5c60-45a7-bc94-8f251f388bfe
# ╠═b68663bb-c948-4609-af81-8c60e54bc0e8
# ╠═920f88d9-24d3-4626-88a9-403f862e4b70
# ╠═3ead108a-f356-4b6b-ad3f-fd8610992175
# ╟─5dcfcd15-e818-4676-927d-786f7aef53f5
# ╟─c3afb1dc-6b77-4e83-af5f-d17eea44bdc8
# ╠═cf348dbb-4b85-483b-8fb9-82a34da082bb
# ╟─f77d7d1f-3279-4915-a6b7-2e861504dd81
# ╟─f49c3d1a-1a89-4ae8-9b1f-7810ca394af1
# ╠═ca258b13-5d4e-4bbd-8f7a-50258d1be3b2
# ╠═9103f75a-873c-4d66-bcc5-b5b71932c3c3
# ╠═2fc14dd3-8f3c-4176-a80d-5f4231f29ba8
# ╟─d1f75886-dc30-4020-af5e-4c3b0dd2149c
# ╟─eb30a1fb-9621-45c0-9acf-92549374cd1e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
