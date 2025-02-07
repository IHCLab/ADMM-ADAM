%=====================================================================
% Programmer: Po-Wei Tang
% E-mail: q38091526@gs.ncku.edu.tw
% Date: June 13, 2022
% -------------------------------------------------------
% Reference:
% ``ADMM-ADAM: A new inverse imaging framework blending the advantages of convex optimization and deep learning,"
% accepted by IEEE Transactions on Geoscience and Remote Sensing, 2021.
%======================================================================
% [X3D_rec,time] = ADMMADAM(X3D_corrupted,mask_3D,X3D_DL)
%======================================================================
% Input
% X3D_corrupted is the corrupted image with dead pixels or stripes, whose dimension is row*col*bands.
% mask_3D is the index set of missing data X3D_corrupted, composed of 0 or 1.
% X3D_DL is the reconstructed image from a weak Adam optimizer-based deep learning solution.
%----------------------------------------------------------------------
% Output
% X3D_rec is the inpainting result whose dimension is the same as X3D_corrupted.
% time is the computation time (in secs).
%========================================================================
function [X3D_rec,time] = ADMMADAM(X3D_corrupted,mask_3D,X3D_DL)
t1 = clock;
%% parameters
N=10; % dimension of the hyperspectral subspace
lambda=0.01; % regularization parameter
mu=1e-3; % ADMM penalty parameter
%% compute S_DL by X_DL
[row, col , bands] = size(X3D_corrupted);
spatial_len=row*col;
X2D_DL = reshape(X3D_DL,[],bands)';
[E_DL] = compute_basis(X3D_DL,N);
S_DL = E_DL'*X2D_DL;
%% compute Sleft(inv(RRT+((mu/2)*INL))) and RPy ( for closed form of S )
mask_2D = reshape(mask_3D,spatial_len,bands)';
nz_idx = sparse([1;zeros(bands,1)]);
M_idx = sparse(kron(mask_2D,nz_idx));
M = M_idx(1:bands^2,:);
PtrpsP_blkdiag = reshape(ndSparse(M),[bands,bands,spatial_len]);
Omega = full(PtrpsP_blkdiag);
RP_blkdiag_tensor= ttm(tensor(Omega),E_DL',1);
RRtrps_tensor= ttm(RP_blkdiag_tensor,E_DL',2);
RP_blkdiag=RP_blkdiag_tensor.data;
RPY= [];
X2D_corrupted = reshape(X3D_corrupted,[],bands)';
for ii = 1:spatial_len
    RPY(:,:,ii) = RP_blkdiag(:,:,ii)*X2D_corrupted(:,ii);
end
RPy = reshape(RPY,[],1);
RRtrps = RRtrps_tensor.data;
I=(mu/2)*eye(N);
block_inv=zeros(size(RRtrps,1),size(RRtrps,2),size(RRtrps,3));
for i=1:size(RRtrps,3)
    block_inv(:,:,i)=inv(reshape(RRtrps(:,:,i),N,[])+I);% inv(E'*Omega*E+(mu/2)*I)
end
Kcell = cellfun(@sparse , num2cell(block_inv,[1,2]) , 'uni', 0 );
S_left=blkdiag(Kcell{:});
%% ADMM iteration
for i = 0:50
    if i==0
        S2D = zeros(N,spatial_len);
        D=zeros(N,spatial_len);
    end
    Z = (1/(mu+lambda))*(lambda*S_DL+mu*(S2D-D)); % update Z
    DELTA = (Z+D);
    delta = reshape(DELTA,[],1);
    s_right = RPy + (mu/2)*delta;
    s = S_left*s_right;
    S2D = reshape(s,[N,spatial_len]); % update S
    D = D - S2D + Z; %update D
end
X2D_rec=E_DL*S2D;
X3D_rec = reshape(X2D_rec',row, col , bands);
time = etime(clock, t1);