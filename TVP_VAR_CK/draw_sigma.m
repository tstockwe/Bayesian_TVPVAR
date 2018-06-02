    % First create capAt, the lower-triangular matrix A(t) with ones on the
    % main diagonal. This is because the vector Atdraw has a draw of only the
    % non-zero and non-one elements of A(t) only.
    capAt = zeros(M*t,M);
    for i = 1:t
        capatemp = eye(M);
        aatemp = Atdraw(:,i);
        ic=1;
        for j = 2:M
            capatemp(j,1:j-1) = aatemp(ic:ic+j-2,1)';
            ic = ic + j - 1;
        end
        capAt((i-1)*M+1:i*M,:) = capatemp;
    end
    
    % yhat is the vector y(t) - Z x B(t) defined previously. Multiply yhat
    % with capAt, i.e the lower triangular matrix A(t). Then take squares
    % of the resulting quantity (saved in matrix y2)
    y2 = [];
    for i = 1:t
        ytemps = capAt((i-1)*M+1:i*M,:)*yhat(:,i);
        y2 = [y2  (ytemps.^2)]; %#ok<AGROW>
    end
    
    yss = log(y2 + 1e-6)';
    for j=1:M
        [Sigtdraw(:,j) , statedraw(:,j)] = SVRW2(yss(:,j),Sigtdraw(:,j),Wdraw(j,:),sigma_prmean(j),sigma_prvar(j,j),1);
    end
    sigt = exp(.5*Sigtdraw);
    
    e2 = Sigtdraw(2:end,:) - Sigtdraw(1:end-1,:);
    W1 = W_prvar + t - p - 1;
    W2 = W_prmean + sum(e2.^2)';
    Winvdraw = gamrnd(W1./2,2./W2);
    Wdraw = 1./Winvdraw;
    