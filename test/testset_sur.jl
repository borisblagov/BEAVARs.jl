
T = 228;
n = 3;
p = 4;
k = n*p + 1
X = rand(T,k);
Xsur = BEAVARs.XSurFormMatrix(X,n);
Xsur_orig, Xsur_CI, X_CI = BEAVARs.SUR_form_dense(X,n);
Xtemp = rand(n,n);
Σt_inv = inv(Xtemp'*Xtemp);
Σ_inv_sp, Σt_LI              = BEAVARs.makeBlkDiag(T*n,n,0,Σt_inv);      # I(T) ⊗ Σ-1 and its indices for update
XtΣ_inv_den                 = zeros(k*n,T*n);                           # will be X' ( I(T) ⊗ Σ-1 )   from page 6 in Chan 2020 LBA
XtΣ_inv                     = zeros(k*n,T*n);                           # will be X' ( I(T) ⊗ Σ-1 )   from page 6 in Chan 2020 LBA
XtΣ_inv_Xden                = zeros(n*k,n*k);                           # will be X' ( I(T) ⊗ Σ-1 ) X from page 6 in Chan 2020 LBA   
XtΣ_inv_X                   = zeros(n*k,n*k);                           # will be X' ( I(T) ⊗ Σ-1 ) X from page 6 in Chan 2020 LBA   


mul!(XtΣ_inv_den,Xsur_orig',Σ_inv_sp);                #  X'*( I(T) ⊗ Σ^{-1} )
mul!(XtΣ_inv,Xsur',Σ_inv_sp);                         #  X'*( I(T) ⊗ Σ^{-1} )

mul!(XtΣ_inv_X,XtΣ_inv_den,Xsur);
mul!(XtΣ_inv_Xden,XtΣ_inv_den,Xsur_orig);


@test Xsur == Xsur_orig                 # test whether the Xsur is defined correctly
@test XtΣ_inv_X ≈ XtΣ_inv_Xden         # test for whether the mul! is implemented correctly
@test XtΣ_inv_den == XtΣ_inv            # test for whether the mul! is implemented correctly for adjoint
