# Results: suite synthetic

Skill: 100 * (1 - CRPS(method) / CRPS(base)), percent, positive is better. Scores are scaled by the seasonal naive error of each node.

## Synthetic grids: skill over base by synthetic_common_sd, all windows

|   synthetic_common_sd |    bu |    ols |   wls_var |   mint_shrink |   prior_shrink |   prior_only |
|----------------------:|------:|-------:|----------:|--------------:|---------------:|-------------:|
|                 0.000 | 4.568 | -7.978 |    -2.687 |         2.371 |          3.819 |        3.188 |
|                 0.350 | 1.551 | -4.741 |    -0.848 |        -3.060 |         -0.332 |       -1.572 |
|                 0.800 | 2.265 | -3.903 |     1.045 |        -5.224 |         -0.597 |       -6.385 |

## Synthetic grids: skill over base by synthetic_regional_sd, all windows

|   synthetic_regional_sd |    bu |    ols |   wls_var |   mint_shrink |   prior_shrink |   prior_only |
|------------------------:|------:|-------:|----------:|--------------:|---------------:|-------------:|
|                   0.000 | 0.414 | -5.148 |    -6.177 |        -7.384 |         -1.506 |       -2.835 |
|                   0.400 | 4.726 | -8.584 |     0.898 |         0.313 |          6.427 |        5.159 |
|                   0.900 | 7.991 | -9.678 |     6.287 |         5.768 |          5.302 |        3.764 |

## Synthetic grids: skill over base by synthetic_regions, all windows

|   synthetic_regions |    bu |    ols |   wls_var |   mint_shrink |   prior_shrink |   prior_only |
|--------------------:|------:|-------:|----------:|--------------:|---------------:|-------------:|
|                   3 | 2.668 | -5.277 |    -2.821 |        -2.302 |          0.404 |       -0.904 |
|                   6 | 2.900 | -6.790 |    -2.269 |        -2.016 |          2.002 |        0.645 |
|                  10 | 2.946 | -7.735 |    -1.881 |        -2.675 |          3.272 |        1.941 |

## Synthetic grids: prior_shrink minus mint_shrink, skill points, common x regional weather

|   synthetic_common_sd |   0.0 |   0.4 |    0.9 |
|----------------------:|------:|------:|-------:|
|                 0.000 | 0.249 | 3.337 | -1.182 |
|                 0.350 | 6.205 | 3.419 | -1.807 |
|                 0.800 | 5.098 | 7.650 |  1.194 |

## Synthetic grids: skill over base by window

|   window |    bu |    ols |   wls_var |   mint_shrink |   prior_shrink |   prior_only |
|---------:|------:|-------:|----------:|--------------:|---------------:|-------------:|
|        7 | 2.862 | -6.563 |    -2.447 |         1.174 |          0.206 |       -0.117 |
|       14 | 2.861 | -6.573 |    -2.430 |        -2.554 |          0.806 |        0.024 |
|       28 | 2.899 | -6.657 |    -2.333 |        -4.229 |          1.554 |        0.346 |
|       91 | 2.962 | -6.641 |    -1.806 |        -3.390 |          5.496 |        2.470 |

## Synthetic grids: Diebold–Mariano prior_shrink vs mint_shrink by window

|   window |   no difference |   ours better |   reference better |
|---------:|----------------:|--------------:|-------------------:|
|        7 |             332 |           146 |                332 |
|       14 |             202 |           289 |                319 |
|       28 |             209 |           307 |                294 |
|       91 |             200 |           278 |                332 |
