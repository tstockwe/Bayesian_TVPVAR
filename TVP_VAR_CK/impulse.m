function response=impulse(By,smat,nstep)
% function response=impulse(By,smat,nstep), C. Sims' code.
% smat is a square matrix of initial shock vectors.  To produce "orthogonalized
% impulse responses" it should have the property that smat'*smat=sigma, where sigma
% is the Var(u(t)) matrix and u(t) is the residual vector.  One way to get such a smat
% is to set smat=chol(sigma).  To get the smat corresponding to a different ordering,
% use smat=chol(P*Sigma*P')*P, where P is a permutation matrix.
% By is a neq x nvar x nlags matrix.  neq=nvar, of course, but the first index runs over 
% equations. In response, the first index runs over variables, the second over 
% shocks (in effect, equations).
[neq,nvar,nlag]=size(By);
response=zeros(nvar,neq,nstep);
response(:,:,1)=smat'; % need lower triangular, last innovation untransformed
for it=2:nstep
   for ilag=1:min(nlag,it-1)
      response(:,:,it)=response(:,:,it)+By(:,:,ilag)*response(:,:,it-ilag);
   end
end
