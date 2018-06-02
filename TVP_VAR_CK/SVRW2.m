% univariate stochastic volatility with random walk transition eq
% Function slightly adapted from Joshua CC Chan's function


function [h, S] = SVRW2(Ystar,h,sig,sigma_prmean,sigma_prvar,TVP_Sigma)

T = length(h);
% normal mixture
pi = [0.0073 .10556 .00002 .04395 .34001 .24566 .2575];
mi = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;  %% means already adjusted!! %%
sigi = [5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261];
sqrtsigi = sqrt(sigi);

% sample S from a 7-point distrete distribution
temprand = rand(T,1);
q = repmat(pi,T,1).*normpdf(repmat(Ystar,1,7),repmat(h,1,7)+repmat(mi,T,1), repmat(sqrtsigi,T,1));
q = q./repmat(sum(q,2),1,7);
S = 7 - sum(repmat(temprand,1,7)<cumsum(q,2),2)+1;

vart = zeros(1,1,T);
yss1 = zeros(T,1);
for i = 1:T
    imix = S(i);
    vart(1,1,i) = sigi(imix);
    yss1(i,1) = Ystar(i,1) - mi(imix);
end

[h,log_lik3] = carter_kohn2(yss1',ones(T,1),vart,sig,1,1,T,sigma_prmean,sigma_prvar,TVP_Sigma*ones(T,1));
