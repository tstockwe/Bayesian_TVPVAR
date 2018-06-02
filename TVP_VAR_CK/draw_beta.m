    % Btdrawc is a draw of the mean VAR coefficients, B(t)
    [Btdrawc,log_lik] = carter_kohn(y,Z,Ht,Qdraw,K,M,t,B_0_prmean,B_0_prvar);
    
    % Accept draw
    Btdraw = Btdrawc;
    % or use the code below to check for stationarity
%     %Now check for the polynomial roots to see if explosive
%     ctemp1 = zeros(M,M*p);
%     counter = [];
%     restviol=0;
%     for i = 1:t;
%         BBtempor = Btdrawc(:,i);
%         BBtempor = reshape(BBtempor,M*p,M)';
%         ctemp1 = [BBtempor; eye(M*(p-1)) zeros(M*(p-1),M)];
%         if max(abs(eig(ctemp1)))>0.9999;
%             restviol=1;
%             counter = [counter ; restviol]; %#ok<AGROW>
%         end
%     end
%     %if they have been rejected keep old draw, otherwise accept new draw 
%     if sum(counter)==0
%         Btdraw = Btdrawc;
%         disp('I found a keeper!');
%     end

    %=====| Draw Q, the covariance of B(t) (from iWishart)
    % Take the SSE in the state equation of B(t)
    Btemp = Btdraw(:,2:t)' - Btdraw(:,1:t-1)';
    sse_2 = zeros(K,K);
    for i = 1:t-1
        sse_2 = sse_2 + Btemp(i,:)'*Btemp(i,:);
    end
    
    % ...and subsequently draw Q, the covariance matrix of B(t)
    Qinv = inv(sse_2 + Q_prmean);
    Qinvdraw = wish(Qinv,t-1+Q_prvar);
    Qdraw = inv(Qinvdraw);  % this is a draw from Q