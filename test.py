import torch

device = 'cuda'

# Uncertainty sigmas, initial sigmas for initialization (but not priors?)
translation_sigma = torch.tensor(0.01, device=device)  # standard deviation of translation [m]
rotation_sigma = torch.tensor(0.01, device=device)  # standard deviation of rotation [rad]
# TODO: given that the values are much larger than 1.0... we should increase this much more...
sigma_idepth = torch.tensor(0.1,
                                 device=device)  # standard deviation of depth [m] (or inverse depth?) [1/m], we don't know the scale anyway...
t_cov = torch.pow(translation_sigma, 2) * torch.eye(3, device=device)
r_cov = torch.pow(rotation_sigma, 2) * torch.eye(3, device=device)
idepth_prior_cov = torch.pow(sigma_idepth, 2)
g_prior_cov = torch.block_diag(r_cov, t_cov)  # GTSAM convention, rotation first, then translation

print(g_prior_cov.shape)
print(idepth_prior_cov.shape)

a = torch.as_tensor([])
print(a)