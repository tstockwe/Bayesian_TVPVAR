% TVP-VAR Time varying structural VAR with constant covariance matrix
% ------------------------------------------------------------------------------------
% This code implements the Homoskedastic TVP-VAR using the Carter and Kohn (1994)
% algorithm for state-space models.
% ************************************************************************************
% The model is:
%
%     Y(t) = B0(t) + B1(t)xY(t-1) + B2(t)xY(t-2) + u(t) 
% 
%  with u(t)~N(0,H).
% The state equation is
%
%            B(t) = B(t-1) + error
%
% where B(t) = [B0(t),B1(t),B2(t)]'.
%
% ************************************************************************************
%   NOTE: 
%      There are references to equations of Primiceri, "Time Varying Structural Vector
%      Autoregressions & Monetary Policy",(2005),Review of Economic Studies 72,821-852
%      for your convenience. The definition of vectors/matrices is also based on this
%      paper.
% ------------------------------------------------------------------------------------

clear all;
clc;
randn('state',sum(100*clock)); %#ok<*RAND>
rand('twister',sum(100*clock));
%----------------------------------LOAD DATA----------------------------------------
% Load Korobilis (2008) quarterly data
load ydata.dat;
load yearlab.dat;

% % Demean and standardize data
% t2 = size(ydata,1);
% stdffr = std(ydata(:,3));
% ydata = (ydata- repmat(mean(ydata,1),t2,1))./repmat(std(ydata,1),t2,1);

Y=ydata;

% Number of observations and dimension of X and Y
t=size(Y,1); % t is the time-series observations of Y
M=size(Y,2); % M is the dimensionality of Y

% Number of factors & lags:
tau = 40; % tau is the size of the training sample
p = 2; % p is number of lags in the VAR part
% ===================================| VAR EQUATION |==============================
% Generate lagged Y matrix. This will be part of the X matrix
ylag = mlag2(Y,p); % Y is [T x M]. ylag is [T x (Mp)]
% Form RHS matrix X_t = [1 y_t-1 y_t-2 ... y_t-k] for t=1:T
ylag = ylag(p+tau+1:t,:);

K = M + p*(M^2); % K is the number of elements in the state vector
% Create Z_t matrix.
Z = zeros((t-tau-p)*M,K);
for i = 1:t-tau-p % tau is size of training sample, so this is all time except lags and training sample
    ztemp = eye(M);
    for j = 1:p        
        xtemp = ylag(i,(j-1)*M+1:j*M);
        xtemp = kron(eye(M),xtemp);
        ztemp = [ztemp xtemp];  %#ok<AGROW>
    end
    Z((i-1)*M+1:i*M,:) = ztemp;
end

% Redefine FAVAR variables y
y = Y(tau+p+1:t,:)';
yearlab = yearlab(tau+p+1:t);
% Time series observations
t=size(y,2);   % t is now 215 - p - tau = 173

%----------------------------PRELIMINARIES---------------------------------
% Set some Gibbs - related preliminaries
nrep = 500;  % Number of replications, originally 5000
nburn = 200;   % Number of burn-in-draws, originally 2000
it_print = 100;  %Print in the screen every "it_print"-th iteration

%========= PRIORS:
% To set up training sample prior a-la Primiceri, use the following subroutine
[B_OLS,VB_OLS,A_OLS,sigma_OLS,VA_OLS]= ts_prior(Y,tau,M,p); % is this a command?

% % Or use uninformative values
% B_OLS = zeros(K,1);
% VB_OLS = eye(K);

%-------- Now set prior means and variances (_prmean / _prvar)
% This is the Kalman filter initial condition for the time-varying
% parameters B(t)
% B_0 ~ N(B_OLS, 4Var(B_OLS))
B_0_prmean = B_OLS;
B_0_prvar = 4*VB_OLS;

% Note that for IW distribution I keep the _prmean/_prvar notation...
% Q is the covariance of B(t)
% Q ~ IW(k2_Q*size(subsample)*Var(B_OLS),size(subsample))
Q_prmean = ((0.01).^2)*tau*VB_OLS;
Q_prvar = tau;

% Sigma is the covariance of the VAR covariance, SIGMA
% Sigma ~ IW(I,M+1)
Sigma_prmean = eye(M);
Sigma_prvar = M+1;

%========= INITIALIZE MATRICES:
% Specify covariance matrices for measurement and state equations
consQ = 0.0001;
Qdraw = consQ*eye(K);
Qchol = sqrt(consQ)*eye(K);
Btdraw = zeros(K,t);
Sigmadraw = 0.1*eye(M);

% Storage matrices for posteriors and stuff
Bt_postmean = zeros(K,t);
Qmean = zeros(K,K);
Sigmamean = zeros(M,M);

%========= IMPULSE RESPONSES:
% Note that impulse response and related stuff involves a lot of storage
% and, hence, put istore=0 if you do not want them
istore = 1;
if istore == 1;
    nhor = 21;     % Impulse response horizon
    shock = diag([zeros(1,M-1) .25]');
    imp75 = zeros(nrep,M,nhor);
    imp81 = zeros(nrep,M,nhor);
    imp96 = zeros(nrep,M,nhor);
    bigj = zeros(M,M*p);
    bigj(1:M,1:M) = eye(M);
end
%----------------------------- END OF PRELIMINARIES ---------------------------

%====================================== START SAMPLING ========================================
%==============================================================================================
tic; % This is just a timer
disp('Number of iterations');

for irep = 1:nrep + nburn    % GIBBS iterations starts here
    % Print iterations
    if mod(irep,it_print) == 0
        disp(irep);toc;
    end
    % -----------------------------------------------------------------------------------------
    %   STEP I: Sample B_t from p(B_t|y,Sigma) (Drawing coefficient states, pp. 844-845)
    % -----------------------------------------------------------------------------------------
    [Btdraw,log_lik] = carter_kohn_hom(y,Z,Sigmadraw,Qdraw,K,M,t,B_0_prmean,B_0_prvar);
    
    
    
    
    Btemp = Btdraw(:,2:t)' - Btdraw(:,1:t-1)';
    sse_2Q = zeros(K,K);
    for i = 1:t-1
        sse_2Q = sse_2Q + Btemp(i,:)'*Btemp(i,:);
    end
 % Drawing Q, based on the Wishart
    Qinv = inv(sse_2Q + Q_prmean);
    Qinvdraw = wish(Qinv,t+Q_prvar);
    Qdraw = inv(Qinvdraw);
    Qchol = chol(Qdraw);
    
    % -----------------------------------------------------------------------------------------
    %   STEP I: Sample Sigma from p(Sigma|y,B_t) which is i-Wishart
    % ----------------------------------------------------------------------------------------
    yhat = zeros(M,t);
    for i = 1:t
        yhat(:,i) = y(:,i) - Z((i-1)*M+1:i*M,:)*Btdraw(:,i);
    end
    
    sse_2S = zeros(M,M);
    for i = 1:t
        sse_2S = sse_2S + yhat(:,i)*yhat(:,i)';
    end
    
    Sigmainv = inv(sse_2S + Sigma_prmean);
    Sigmainvdraw = wish(Sigmainv,t+Sigma_prvar);
    Sigmadraw = inv(Sigmainvdraw);
    Sigmachol = chol(Sigmadraw);
    
    %----------------------------SAVE AFTER-BURN-IN DRAWS AND IMPULSE RESPONSES -----------------
    if irep > nburn;
        % Save only the means of B(t), Q and SIGMA. Not memory efficient to
        % store all draws (at least for B(t) which is large). If you want to
        % store all draws, it is better to save them in a file at each iteration.
        % Use the MATLAB command 'save' (type 'help save' in the command window
        % for more info)
        Bt_postmean = Bt_postmean + Btdraw;
        Qmean = Qmean + Qdraw;
        Sigmamean = Sigmamean + Sigmadraw;
         
        if istore==1;
            % Impulse response analysis. Note that Htsd contains the
            % structural error cov matrix
            % Set up things in VAR(1) format as in Lutkepohl (2005) page 51
            biga = zeros(M*p,M*p);
            for j = 1:p-1
                biga(j*M+1:M*(j+1),M*(j-1)+1:j*M) = eye(M);
            end

            for i = 1:t %Get impulses recurssively for each time period
                bbtemp = Btdraw(M+1:K,i);  % get the draw of B(t) at time i=1,...,T  (exclude intercept)
                splace = 0;
                for ii = 1:p
                    for iii = 1:M
                        biga(iii,(ii-1)*M+1:ii*M) = bbtemp(splace+1:splace+M,1)';
                        splace = splace + M;
                    end
                end
                
                % ------------Identification code I:                
                % st dev matrix for structural VAR
                shock = Sigmachol';   % First shock is the Cholesky of the VAR covariance
                diagonal = diag(diag(shock));
                shock = inv(diagonal)*shock;    % Unit initial shock 
                
                % Now get impulse responses for 1 through nhor future periods
                impresp = zeros(M,M*nhor); % matrix to store initial response at each period
                impresp(1:M,1:M) = shock;  % First shock is the Cholesky of the VAR covariance
                bigai = biga;
                for j = 1:nhor-1
                    impresp(:,j*M+1:(j+1)*M) = bigj*bigai*bigj'*shock;
                    bigai = bigai*biga;
                end

                % Only for specified periods
                if yearlab(i,1) == 1975.00;   % 1975:Q1
                    impf_m = zeros(M,nhor);
                    jj=0;
                    for ij = 1:nhor
                        jj = jj + M;    % restrict to the M-th equation, the interest rate
                        impf_m(:,ij) = impresp(:,jj);
                    end
                    imp75(irep-nburn,:,:) = impf_m; % store draws of responses
                end
                if yearlab(i,1) == 1981.50;   % 1981:Q3
                    impf_m = zeros(M,nhor);
                    jj=0;
                    for ij = 1:nhor
                        jj = jj + M;    % restrict to the M-th equation, the interest rate
                        impf_m(:,ij) = impresp(:,jj);
                    end
                    imp81(irep-nburn,:,:) = impf_m;  % store draws of responses
                end
                if yearlab(i,1) == 1996.00;   % 1996:Q1
                    impf_m = zeros(M,nhor);
                    jj=0;
                    for ij = 1:nhor
                        jj = jj + M;    % restrict to the M-th equation, the interest rate
                        impf_m(:,ij) = impresp(:,jj);
                    end
                    imp96(irep-nburn,:,:) = impf_m;  % store draws of responses
                end
            end %END geting impulses for each time period 
        end %END the impulse response calculation section   
    end % END saving after burn-in results 
end %END main Gibbs loop (for irep = 1:nrep+nburn)
clc;
toc; % Stop timer and print total time
%=============================GIBBS SAMPLER ENDS HERE==================================
Bt_postmean = Bt_postmean./nrep;   % Posterior mean of B(t) (VAR regression coeff.)
Qmean = Qmean./nrep;               % Posterior mean of Q (covariance of B(t))
Sigmamean = Sigmamean./nrep;       % Posterior mean of SIGMA (VAR covariance matrix)

if istore == 1 
    qus = [.16, .5, .84];
    imp75XY=squeeze(quantile(imp75,qus));
    imp81XY=squeeze(quantile(imp81,qus));
    imp96XY=squeeze(quantile(imp96,qus));
    
    % Plot impulse responses
    figure       
    set(0,'DefaultAxesColorOrder',[0 0 0],...
        'DefaultAxesLineStyleOrder','--|-|--')
    subplot(3,3,1)
    plot(1:nhor,squeeze(imp75XY(:,1,:)))
    title('Impulse response of inflation, 1975:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)
    subplot(3,3,2)
    plot(1:nhor,squeeze(imp75XY(:,2,:)))
    title('Impulse response of unemployment, 1975:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,3)
    plot(1:nhor,squeeze(imp75XY(:,3,:)))
    title('Impulse response of interest rate, 1975:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,4)
    plot(1:nhor,squeeze(imp81XY(:,1,:)))
    title('Impulse response of inflation, 1981:Q3')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,5)
    plot(1:nhor,squeeze(imp81XY(:,2,:)))
    title('Impulse response of unemployment, 1981:Q3')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,6)
    plot(1:nhor,squeeze(imp81XY(:,3,:)))
    title('Impulse response of interest rate, 1981:Q3')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,7)
    plot(1:nhor,squeeze(imp96XY(:,1,:)))
    title('Impulse response of inflation, 1996:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,8)
    plot(1:nhor,squeeze(imp96XY(:,2,:)))
    title('Impulse response of unemployment, 1996:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)
    subplot(3,3,9)
    plot(1:nhor,squeeze(imp96XY(:,3,:)))
    title('Impulse response of interest rate, 1996:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)
end

disp('             ')
disp('To plot impulse responses, use:         plot(1:nhor,squeeze(imp75XY(:,VAR,:)))           ')
disp('             ')
disp('where VAR=1 for impulses of inflation, VAR=2 for unemployment and VAR=3 for interest rate')



