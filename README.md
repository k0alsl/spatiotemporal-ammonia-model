# spatiotemporal-ammonia-model

[![DOI](https://zenodo.org/badge/974489048.svg)](https://doi.org/10.5281/zenodo.15426285)

This repository is a codebase for applying a spatiotemporal modeling approach to estimate ambient gas/aerosol concentrations. The model assumes that concentrations are largely explained by the combination of temporal basis functions with the amplitudes determined by land use regression (LUR).

The main codes were written in Julia, where we applied the modeling approach for ammonia (NH<sub>3</sub>) concentrations in Champaign, IL, USA for approximately an year (2022-2023). The data required to run the code is also included in this repository.

- `src/Project.toml` and `src/Manifest.toml` - Files that define the required environment.
- `src/base.jl` - A Julia script where utility functions are defined.
- `src/stmodel_nh3_cu.jl` - A Julia script where input data is imported; a model is constructed and evaluated; and finally fine-scale predictions are made using the model.
- `input/sites.csv` - Monitoring site information including site ID, type, and location.
- `input/nh3_transformed_imputed.csv` - Pretreated monitoring data.
- `input/cov_transformed.csv` - Pretreated geospatial covariates for monitoring sites.

For details, please refer to following articles: Kim and Tessum, 2026; [Keller et al., 2015](https://doi.org/10.1289/ehp.1408145).
