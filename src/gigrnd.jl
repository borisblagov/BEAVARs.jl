"""
    gigrnd(P,a,b, sampleSize)

    This is an LLM generated function
"""
function gigrnd(P, a, b, sampleSize)
    # Helper functions
    psi(x, alpha, lambda) = -alpha * (cosh(x) - 1) - lambda * (exp(x) - x - 1)
    dpsi(x, alpha, lambda) = -alpha * sinh(x) - lambda * (exp(x) - 1)
    
    function g(x, sd, td, f1, f2)
        a_val = 0.0
        b_val = 0.0
        c_val = 0.0
        if (x >= -sd) && (x <= td)
            a_val = 1.0
        elseif x > td
            b_val = f1
        elseif x < -sd
            c_val = f2   
        end
        return a_val + b_val + c_val
    end

    ## Setup -- we sample from the two parameter version of the GIG(alpha,omega)
    lambda = P
    omega = sqrt(a * b)
    swap = false
    if lambda < 0
        lambda = lambda * -1
        swap = true
    end
    alpha = sqrt(omega^2 + lambda^2) - lambda

    ## Find t
    x = -psi(1.0, alpha, lambda)
    t = 0.0
    if (x >= 0.5) && (x <= 2.0)
        t = 1.0
    elseif x > 2.0
        t = sqrt(2.0 / (alpha + lambda))
    elseif x < 0.5
        t = log(4.0 / (alpha + 2.0 * lambda))     
    end

    ## Find s
    x = -psi(-1.0, alpha, lambda)
    s = 0.0
    if (x >= 0.5) && (x <= 2.0)
        s = 1.0
    elseif x > 2.0
        s = sqrt(4.0 / (alpha * cosh(1.0) + lambda))
    elseif x < 0.5
        s = min(1.0 / lambda, log(1.0 + 1.0 / alpha + sqrt(1.0 / alpha^2 + 2.0 / alpha)))
    end

    ## Generation
    eta = -psi(t, alpha, lambda)
    zeta = -dpsi(t, alpha, lambda)
    theta = -psi(-s, alpha, lambda)
    xi = dpsi(-s, alpha, lambda)
    p = 1.0 / xi
    r = 1.0 / zeta
    td = t - r * eta
    sd = s - p * theta
    q = td + sd
    
    X = zeros(sampleSize)
    for sample in 1:sampleSize
        done = false
        while !done
            U = rand() 
            V = rand() 
            W = rand()
            
            if U < (q / (p + q + r))
                X[sample] = -sd + q * V
            elseif U < ((q + r) / (p + q + r))
                X[sample] = td - r * log(V)
            else
                X[sample] = -sd + p * log(V)
            end
            
            ## Are we done?
            f1 = exp(-eta - zeta * (X[sample] - t))
            f2 = exp(-theta + xi * (X[sample] + s))
            if (W * g(X[sample], sd, td, f1, f2)) <= exp(psi(X[sample], alpha, lambda))
                done = true
            end
        end
    end

    ## Transform X back to the three parameter GIG(p,a,b)
    X = exp.(X) .* (lambda / omega + sqrt(1.0 + (lambda / omega)^2))
    if swap
        X = 1.0 ./ X
    end
    X = X ./ sqrt(a / b)
    
    return X
end