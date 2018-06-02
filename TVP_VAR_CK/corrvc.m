function [output] = corrvc(vc)
% corrvc
% 
% Purpose:    Computes a correlation matrix from a
%              variance-covariance matrix.
%              
% Format:     cx = corrvc(vc);
% 
% Input:      vc    KxK variance-covariance matrix (of data or parameters).
% 
% Output:     cx    KxK correlation matrix.


% Check for complex input */
if isreal(vc) == 0;       
    error('ERROR: Not implemented for complex arguments.')
end
std = sqrt(diag(vc));
   
% if (type(vc) == 6);
     output = vc./(std*std');
% elseif (type(vc) == 21);       
%     dims = getdims(vc);
%     if (dims < 3);
%         vc = arraytomat(vc);
%         std = arraytomat(std);
%         output = mattoarray(vc./(std.*std'));
%     else
%         torders = seqa(1,1,dims-2);
%         torders = torders||dims||dims-1;
%         output = vc./(std.*atranspose(std,torders));
%     end
% else
%     error('ERROR: Type mismatch.')
% end