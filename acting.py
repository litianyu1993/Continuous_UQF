from Monte_Carlo_integral import MC_integral
from CWFA_AO import CWFA_AO
import torch
from Encoder import Encoder
from Decoder_learning import get_decoder
from scipy.optimize import linprog

def convert_wfa_to_uqf(cwfa_ao, **option):
    action_encoder = cwfa_ao.action_encoder
    obs_encoder = cwfa_ao.obs_encoder

    option_default = {
        'num_examples': 100000,
        'range': [0, 1],
        'input_dim': action_encoder.input_dim
    }
    option = {**option_default, **option}

    int_action_encoder = MC_integral(action_encoder, **option).reshape(-1, )

    option['input_dim'] = obs_encoder.input_dim
    int_obs_encoder = MC_integral(obs_encoder, **option).reshape(-1, )

    A_tilde = torch.einsum('i, j, kijl->kl', int_action_encoder, int_obs_encoder, cwfa_ao.A)
    #print((torch.eye(A_tilde.shape[0]) - A_tilde).shape, cwfa_ao.Omega.shape, A_tilde.shape)
    Omega_tilde = torch.inverse(torch.eye(A_tilde.shape[0]) - A_tilde) @ cwfa_ao.Omega

    cwfa_option = {
        'alpha': cwfa_ao.alpha,
        'A': cwfa_ao.A,
        'Omega': Omega_tilde,
        'random_init': False
    }

    uqf = CWFA_AO(cwfa_ao.action_encoder, cwfa_ao.obs_encoder, **cwfa_option)
    next_A = torch.einsum('j, kijl->kil', int_obs_encoder, cwfa_ao.A)
    return uqf, next_A



def UQF_agent(uqf, next_A, decoder, history, **option):
    option_default = {
        'range': (0, 1)
    }
    option = {**option_default, **option}
    planning_vec = uqf.planning(history, next_A).detach().numpy().ravel()
    best_a_encoded = linprog(-planning_vec, bounds=option['range']).x
    best_a = decoder(best_a_encoded)
    return best_a


if __name__ == '__main__':

    decoder_option = {
        'sample_size_train': 10000,
        'sample_size_vali': 1000,
        'lr': 0.001,
        'epochs': 10,
        'gamma': 1,
        'step_size': 500,
        'batch_size': 256
    }

    rank = 5
    input_dim = 3
    encoded_dim = 5
    hidden_units = [5]
    print('here')

    encoder_option = {
        'input_dim': input_dim,
        'hidden_units': hidden_units,
        'out_dim': encoded_dim,
        'final_activation': torch.nn.Tanh(),
        'inner_activation': torch.nn.LeakyReLU(inplace=False)
    }

    action_encoder = Encoder(**encoder_option)
    obs_encoder = Encoder(**encoder_option)
    alpha = torch.rand(rank)
    A = torch.rand([rank, encoded_dim, encoded_dim, rank])
    Omega = torch.rand([rank])
    cwfa_option = {
        'alpha': alpha,
        'A':A,
        'Omega': Omega,
        'random_init': False
    }
    cwfa = CWFA_AO(action_encoder, obs_encoder, **cwfa_option)

    uqf, next_A = convert_wfa_to_uqf(cwfa_ao=cwfa)
    action_decoder = get_decoder(action_encoder, **decoder_option)

    actions = torch.normal(0, 1, [1, 5, 3])
    obss = torch.normal(0, 1, [1, 5, 3])
    agent_option = {
        'range': (0, 1)
    }
    print(UQF_agent(uqf, next_A, action_decoder, [actions, obss], **agent_option))


