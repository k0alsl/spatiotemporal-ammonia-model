# spatiotemporal-ammonia-model

This repository is a codebase for applying a spatiotemporal modeling approach to estimate ambient gas/aerosol concentrations. The model assumes that concentrations are largely explained by the combination of temporal basis functions with the amplitudes determined by land use regression (LUR).

The main codes were written in Julia on a Pluto notebook, where we applied the modeling approach for ammonia (NH<sub>3</sub>) concentrations in Champaign, IL, USA in years 2022-2023. The data required to run the code is also included in this repository.
The monitoring data (including data retrieved from AMoN) and the covariates are also included in this repository.

- `stmodel_nh3_pluto.jl` - A Pluto notebook where input data is imported and pretreated; a model is constructed and evaluated; and finally fine-scale predictions are made using the model.
- `nh3.csv` - NH<sub>3</sub> monitoring data including data retrieved from [the National Ammonia Monitoring Network (AMoN)](https://nadp.slh.wisc.edu/networks/ammonia-monitoring-network/).
- `cov_site.csv` - Land use-related covariates computed for our monitoring sites.
- `cov_roi.csv` - Land use-related covariates computed for a 7 km-by-7 km region at 30-m resolution.

For details, please look into the Pluto notebook, or refer to following articles: Kim and Tessum, 2025; [Keller et al., 2015](https://doi.org/10.1289/ehp.1408145).
