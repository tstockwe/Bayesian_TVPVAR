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

                % ------------Identification code:                
                % St dev matrix for structural VAR
                Hsd = Htsd((i-1)*M+1:i*M,1:M);   % First shock is the Cholesky of the VAR covariance
                diagonal = diag(diag(Hsd));
                Hsd = inv(diagonal)*Hsd;    % Unit initial shock
                
                % Now get impulse responses for 1 through nhor future periods
                impresp = zeros(M,M*nhor);
                impresp(1:M,1:M) = Hsd; % First shock is the Cholesky of the VAR covariance
                bigai = biga;
                for j = 1:nhor-1
                    impresp(:,j*M+1:(j+1)*M) = bigj*bigai*bigj'*Hsd;
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