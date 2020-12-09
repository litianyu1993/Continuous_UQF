from torch import nn
from Monte_Carlo_integral import MC_integral
import torch
from scipy.optimize import linprog

class UQF(nn.Module):

    def __init__(self, alpha, A, Omega, action_encoder, action_decoder, obs_encoder,
                 obs_encoder_intg, encoded_action_range, action_range):

        super(UQF, self).__init__()
        self.alpha = alpha
        self.A = A
        self.Omega = Omega
        self.action_encoder = action_encoder
        self.obs_encoder = obs_encoder
        self.plan_Omega = torch.einsum('ijkl, k, l -> ij', self.A, obs_encoder_intg, self.Omega)
        self.action_decoder = action_decoder
        self.action_range = action_range
        self.encoded_action_range = encoded_action_range


    def update_dynamics(self, action, obs, belief):
        action_encoded = self.action_encoder(action)
        obs_encoded = self.obs_encoder(obs)
        return torch.einsum('ijkl, nj, nk, ni -> nl', self.A, action_encoded, obs_encoded, belief)

    def agent(self, belief):
        c = torch.einsum('j, jk -> k', belief, self.plan_Omega)
        res = linprog(c, bounds=self.encoded_action_range)
        action = self.action_decoder(res.x)
        return action




class Encoder(nn.Module):
    def __init__(self, cwfa_encoder, cwfa):
        super(Encoder, self).__init__()
        self.encoder = cwfa_encoder
        self.cwfa = cwfa

    def forward(self, x):
        return self.cwfa.encode_FC(self.encoder, x)


def convert_CWFA_to_UQF(cwfa_ao):
    action_encoder = Encoder(cwfa_ao.action_encoder, cwfa_ao)
    obs_encoder = Encoder(cwfa_ao.obs_encoder, cwfa_ao)
    intg_option = {
        'num_examples': 100000,
        'range': [-1, 1],
        'input_dim': cwfa_ao.dim_o
    }
    obs_encoder_intg = MC_integral(obs_encoder, **intg_option)
    intg_option['input_dim'] = cwfa_ao.dim_a
    action_encoder_intg = MC_integral(action_encoder, **intg_option)

    tmp_A = torch.einsum('ijkl, j, k -> il', cwfa_ao.A, obs_encoder_intg, action_encoder_intg)
    Omega = torch.inverse(torch.eye(cwfa_ao.A.shape[0]) - tmp_A).dot(cwfa_ao.Omega)
    return UQF(cwfa_ao.alpha, cwfa_ao.A, Omega, action_encoder, obs_encoder, obs_encoder_intg)



