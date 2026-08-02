include("src/base.jl")



## Import data ##
sites = CSV.read("input/sites.csv", DataFrame)
obs_df = CSV.read("input/nh3_transformed_imputed.csv", DataFrame)
covs = CSV.read("input/cov_transformed.csv", DataFrame)

## Extract basic elements from data ##
dates = obs_df.date
obs = obs_df[:,2:end] |> Matrix
# keepidx = ismissing.(obs) .== false
# ntime, nsite = size(obs)
cov_mat = covs[:,2:end] |> Matrix



## Cross-validation over number of time trends
smooth_intensity = 0.5  # 0-1.5 allowed
cv_ntimetrends(obs[:,sites.longterm], smooth_intensity)

## Extract time trends ##
num_timetrends = 2
smooth_intensity = 0.5  # 0-1.5 allowed
timetrends = calc_timetrends(obs[:,sites.longterm], num_timetrends, smooth_intensity)
# Invert time trends if needed
timetrends[:,2] .*= -1

## Time trend fitting ##
fitted_betas = regress_beta(obs, timetrends)



## Cross-validation over number of PLS components ##
beta_means = mean(fitted_betas, dims=2)
cv_npls(cov_mat, fitted_betas .- beta_means)

## PLS regression ##
num_pls_factors = 8
beta_means = mean(fitted_betas, dims=2)
pls_betas, pls_mach = pls_regress_beta(cov_mat, cov_mat, fitted_betas .- beta_means, num_pls_factors)
pls_betas .+= beta_means



## Krig beta fields ##
beta_range, beta_sill, beta_nugget= 1100.0, 0.0035, 0.0019
geoms = Point.(sites.x,sites.y)
beta_resids = fitted_betas .- pls_betas
pred_betas = pls_betas .+ krig_beta_resid(beta_resids, geoms, geoms, beta_range, beta_sill, beta_nugget)



## Krig residuals ##
nu_range, nu_sill, nu_nugget = 5600.0, 0.0018, 0.00068
geoms = Point.(sites.x,sites.y)
pred_mus = calc_mu(timetrends, pred_betas)
resids = obs .- pred_mus
pred_nus = krig_nu(resids, geoms, geoms, nu_range, nu_sill, nu_nugget)



## Total predictions ##
pred_ys = pred_mus .+ pred_nus
pred_df = hcat(DataFrame(date=dates), DataFrame(pred_ys, sites.id))



## Collect parameters ##
stmod_params = (
    geometry = geoms,
    timetrends = timetrends,
    pls_mach = pls_mach,
    beta_means = beta_means,
    beta_resids = beta_resids,
    beta_krig = (beta_range, beta_sill, beta_nugget),
    resids = resids,
    nu_krig = (nu_range, nu_sill, nu_nugget),
);

## Prediction at a location ##
stmod_predict(stmod_params, geoms[1], cov_mat[1,:])
