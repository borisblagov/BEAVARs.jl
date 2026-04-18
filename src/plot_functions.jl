function fanChart(toPlot_mat)
    prct_dim = length(size(toPlot_mat)); # the dimension with the draws (always the last)
    if prct_dim == 3
        n=size(toPlot_mat,2);
    elseif prct_dim ==2
        n = 1;
    end
    Yfor_low1 = percentile_mat(toPlot_mat,0.05,dims=prct_dim);
    Yfor_low = percentile_mat(toPlot_mat,0.16,dims=prct_dim);
    Yfor_med = percentile_mat(toPlot_mat,0.5,dims=prct_dim);
    Yfor_hih = percentile_mat(toPlot_mat,0.84,dims=prct_dim);
    Yfor_hih1 = percentile_mat(toPlot_mat,0.95,dims=prct_dim);

    if n < 4
        lout = (n,1)
    elseif n == 4
        lout = (2,2)
    elseif n>4 
        lout = (convert(Int,ceil(n/4)),4)
    # elseif n>11
    #     lout = (convert(Int,ceil(n/4)),4)
    end

    p_hand = plot(layout=lout)
    for ik in 1:n
        plot!(p_hand,Yfor_med[:,ik],w=0;ribbon=(Yfor_med[:,ik]-Yfor_low1[:,ik],Yfor_hih1[:,ik]-Yfor_med[:,ik]),fillalpha = 0.1,color=1,legend=false,subplot=ik) #,title=varList[ik]
        plot!(p_hand,Yfor_med[:,ik],w=2;ribbon = (Yfor_med[:,ik]-Yfor_low[:,ik],Yfor_hih[:,ik]-Yfor_med[:,ik]),fillalpha=0.05,color=1,subplot=ik)
        
    end
    display(p_hand)
    return Yfor_low1, Yfor_low, Yfor_med, Yfor_hih, Yfor_hih1
end



function fanChart(toPlot_mat,dates::Array{Date, 1})
    prct_dim = length(size(toPlot_mat)); # the dimension with the draws (always theh last)
    if prct_dim == 3
        n=size(toPlot_mat,2);
    elseif prct_dim ==2
        n = 1;
    end
    Yfor_low1 = percentile_mat(toPlot_mat,0.05,dims=prct_dim);
    Yfor_low = percentile_mat(toPlot_mat,0.16,dims=prct_dim);
    Yfor_med = percentile_mat(toPlot_mat,0.5,dims=prct_dim);
    Yfor_hih = percentile_mat(toPlot_mat,0.84,dims=prct_dim);
    Yfor_hih1 = percentile_mat(toPlot_mat,0.95,dims=prct_dim);

    if n < 4
        lout = (n,1)
    elseif n == 4
        lout = (2,2)
    elseif n>4 
        lout = (convert(Int,ceil(n/4)),4)
    # elseif n>11
    #     lout = (convert(Int,ceil(n/4)),4)
    end

    p_hand = plot(layout=lout)
    for ik in 1:n
        plot!(p_hand,dates,Yfor_med[:,ik],w=0;ribbon=(Yfor_med[:,ik]-Yfor_low1[:,ik],Yfor_hih1[:,ik]-Yfor_med[:,ik]),fillalpha = 0.1,color=1,legend=false,subplot=ik) #,title=varList[ik]
        plot!(p_hand,dates,Yfor_med[:,ik],w=2;ribbon = (Yfor_med[:,ik]-Yfor_low[:,ik],Yfor_hih[:,ik]-Yfor_med[:,ik]),fillalpha=0.05,color=1,subplot=ik)
        
    end
    display(p_hand)
    return Yfor_low1, Yfor_low, Yfor_med, Yfor_hih, Yfor_hih1
end


"""
    forecast_plot(fcast_strct::VARForecast)

The function generates forecast plots from the VARForecast structure. 
The function uses the 3D array of forecasts (Yfor3D) and the variable names (var_list) to create a plot for each variable, showing the median forecast along with confidence intervals based on the percentiles of the forecast distribution. 
The layout of the plots is determined by the number of variables being forecasted.

"""
function forecast_plot(fcast_strct::VARForecast)
    @unpack Yfor3D, data_tab, var_list = fcast_strct
    n = size(Yfor3D,2);
    Yfor_low1 = percentile_mat(Yfor3D,0.05,dims=3);
    Yfor_low = percentile_mat(Yfor3D,0.16,dims=3);
    Yfor_med = percentile_mat(Yfor3D,0.5,dims=3);
    Yfor_hih = percentile_mat(Yfor3D,0.84,dims=3);
    Yfor_hih1 = percentile_mat(Yfor3D,0.95,dims=3);

    if n < 4
        lout = (n,1)
    elseif n == 4
        lout = (2,2)
    elseif n>4 
        lout = (convert(Int,ceil(n/4)),4)
    # elseif n>11
    #     lout = (convert(Int,ceil(n/4)),4)
    end

    p = plot(layout=lout)
    for ik in 1:n
        plot!(p,Yfor_med[:,ik],w=0;ribbon=(Yfor_med[:,ik]-Yfor_low1[:,ik],Yfor_hih1[:,ik]-Yfor_med[:,ik]),fillalpha = 0.1,color=1,legend=false,subplot=ik,title=var_list[ik])
        plot!(p,Yfor_med[:,ik],w=2;ribbon = (Yfor_med[:,ik]-Yfor_low[:,ik],Yfor_hih[:,ik]-Yfor_med[:,ik]),fillalpha=0.05,color=1,subplot=ik)
        
    end
    display(p)

end