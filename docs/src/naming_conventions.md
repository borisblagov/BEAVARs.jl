# Naming conventions

In most of my codes I thought that I try to use similar naming scheme for various objects until I decided to write this down and realised how much of a mess it is. In order to try to have cleaner code I am devising this scheme here and will see if I can implement it. 

## Arrays
`_mat`: two dimensional matrix
`_3dmat`: three dimensional version of the two-dimensional object _mat.
`_save`:  also often three dimensional matrix, used for saving Bayesian draws
`_ta`: a `TimeArray`
`_tab`: a `Table`, often actually also `TimeArray`

## Letters
`T´: time periods
`Tobs´: time periods when accounting for lags
`p`: number of lags
`n`: number of variables
`n_suffix`: the total number of the thing in the suffix, e.g. `n_miss` is number of missing variables, `n_save` is the number of saved draws