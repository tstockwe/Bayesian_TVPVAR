function [bdraw,log_lik] = carter_kohn(y,Z,Ht,Qt,m,p,t,B0,V0)
% Carter and Kohn (1994), On Gibbs sampling for state space models.

% Kalman Filter
bp = B0;
Vp = V0;
bt = zeros(t,m);
Vt = zeros(m^2,t);
log_lik = 0;
for i=1:t
    R = Ht((i-1)*p+1:i*p,:);
    H = Z((i-1)*p+1:i*p,:);
    cfe = y(:,i) - H*bp;   % conditional forecast error
    f = H*Vp*H' + R;    % variance of the conditional forecast error
    inv_f = inv(f);
    %log_lik = log_lik + log(det(f)) + cfe'*inv_f*cfe;
    btt = bp + Vp*H'*inv_f*cfe;
    Vtt = Vp - Vp*H'*inv_f*H*Vp;
    if i < t
        bp = btt;
        Vp = Vtt + Qt;
    end
    bt(i,:) = btt';
    Vt(:,i) = reshape(Vtt,m^2,1);
end

% draw Sdraw(T|T) ~ N(S(T|T),P(T|T))
bdraw = zeros(t,m);
bdraw(t,:) = mvnrnd(btt,Vtt,1);

% Backward recurssions
for i=1:t-1
    bf = bdraw(t-i+1,:)';
    btt = bt(t-i,:)';
    Vtt = reshape(Vt(:,t-i),m,m);
    f = Vtt + Qt;
    inv_f = inv(f);
    cfe = bf - btt;
    bmean = btt + Vtt*inv_f*cfe;
    bvar = Vtt - Vtt*inv_f*Vtt;
    bdraw(t-i,:) = mvnrnd(bmean,bvar,1); %bmean' + randn(1,m)*chol(bvar);
end
bdraw = bdraw';