# spatiotemporal-ammonia-model

[![DOI](https://zenodo.org/badge/974489048.svg)](https://doi.org/10.5281/zenodo.15426285)

This repository is a codebase for applying a spatiotemporal modeling approach to estimate ambient gas/aerosol concentrations. The model assumes that concentrations are largely explained by the combination of temporal basis functions with the amplitudes determined by land use regression (LUR).

The main codes were written in Julia on a Pluto notebook, where we applied the modeling approach for ammonia (NH<sub>3</sub>) concentrations in Champaign, IL, USA for approximately an year (2022-2023). The data required to run the code is also included in this repository.

- `stmodel_nh3_pluto.jl` - A Pluto notebook where input data is imported and pretreated; a model is constructed and evaluated; and finally fine-scale predictions are made using the model.
- `nh3.csv` - NH<sub>3</sub> monitoring data including data retrieved from [the National Ammonia Monitoring Network (AMoN)](https://nadp.slh.wisc.edu/networks/ammonia-monitoring-network/).
- `cov_site.csv` - Site information (site name, type, and location) and land use-related covariates computed for our monitoring sites.
- `grid.csv` - Modeling grid for a 7 km-by-7 km region at 30-m resolution.
- `cov_roi.csv` - Land use-related covariates computed for the modeling grid. This file is offered in a compressed format (.zip) due to its relatively large size (37 MB), so needs unzipping before use.

For details, please look into the Pluto notebook, or refer to following articles: Kim and Tessum, 2025; [Keller et al., 2015](https://doi.org/10.1289/ehp.1408145).
