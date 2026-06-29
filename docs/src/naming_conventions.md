# Naming conventions

In most of my codes I try to use similar naming scheme for various objects. While I do not follow it extremely rigorously I try to keep to those conventions more often than not (I also sometimes get annoyed when similar papers use different conventions, e.g. `N`  often denots the number of variables, while in a Global VARs people use `N` for number of units/countries and `G` for number of variables...

## Arrays
`_mat`: two dimensional matrix
`_3dmat`: three dimensional version of the two-dimensional object _mat.
`_save`:  also often three dimensional matrix, used for saving Bayesian draws
`_ta`: a `TimeArray`

## Letters
`T´: time periods
`Tobs´: time periods when accounting for lags
`p`: number of lags
`n`: number of variables
`n_suffix`: the total number of the thing in the suffix, e.g. `n_miss` is number of missing variables, `n_save` is the number of saved draws