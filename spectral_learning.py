import numpy as np
import tt
from Hankel import Encoder, Hankel, fit_hankel
import gym
from gradien_descent import train, validate
from preprocess import get_dataset, get_data_generator, construct_KDE, get_data_loaders, get_kde
import torch
import pickle
from KDE import load_kde
from CWFA_AO import CWFA_AO
from torch import optim
from torch.optim.lr_scheduler import StepLR
def TT_product(a_cores, b_cores):
  """
  performs the TT chain of product between the two set of cores:
  -- a1 -- a2 -- a3 -- ...
     |     |     |
  -- b1 -- b2 -- b3 -- ...
  """
  for i,(c1,c2) in enumerate(zip(a_cores,b_cores)):
    if i == 0:
      res = np.tensordot(c1,c2,(1,1)).transpose([0,2,1,3])
    else:
      res = np.tensordot(res,c1,(2,0))
      res = np.tensordot(res,c2,((2,3),(0,1)))
  return res

def _rightorth(a, b):
    """
    right orthonormalisation of core a. After this we have
    np.tensordot(a,a,axes=((0,1),(0,1))) = I
    while
    np.tensordot(a,b,axes=(2,0)
    remains unchanged
    """
    adim = np.array(a.shape)
    bdim = np.array(b.shape)
    #print(adim, bdim)
    cr = np.reshape(a, (adim[0]*adim[1], adim[2]), order='F')
    cr2 = np.reshape(b, (bdim[0],bdim[1]*bdim[2]), order='F')
    [cr, R] = np.linalg.qr(cr)
    #print(R.shape, cr2.shape, cr.shape)
    cr2 = np.dot(R, cr2)

    adim[2] = cr.shape[1]
    bdim[0] = cr2.shape[0]
    a = np.reshape(cr, adim, order='F')
    b = np.reshape(cr2, bdim, order='F')
    return a,b

def _leftorth(a,b):
    """
    left orthonormalisation of core a. After this we have
    np.tensordot(b,b,axes=((1,2),(1,2))) = I
    while
    np.tensordot(a,b,axes=(2,0)
    remains unchanged
    """

    adim = np.asarray(a.shape)
    bdim = np.asarray(b.shape)
    cr = np.reshape(b, (bdim[0],bdim[1]*bdim[2]),order='F').T
    [cr, R] = np.linalg.qr(cr)
    cr2 = np.reshape(a, (adim[0]*adim[1],adim[2]),order='F').T
    cr2 = np.dot(R, cr2)

    adim[2] = cr2.T.shape[1]
    bdim[0] = cr.T.shape[0]

    a = np.reshape(cr2.T, adim, order='F')
    b = np.reshape(cr.T, bdim, order='F')

    return a,b

def _orthcores(a,dir='right'):
    """
    right (resp. left) orthonormalize all cores of a except for
    the last (resp. first) one
    """
    if isinstance(a,list):
        d = len(a)
        ry = [X.shape[0] for X in a] + [1]
        L = a
        #a = tt.vector.from_list(L)
    elif isinstance(a,tt.vector):
        d = a.d
        ry = a.r
        L = tt.vector.to_list(a)
    else:
        raise NotImplementedError()

    if dir=='right':
        for i in range(d - 1):
            L[i:i+2] = _rightorth(L[i], L[i + 1])
            ry[i + 1] = L[i].shape[2]
    elif dir=='left':
        for i in range(d-1,0,-1):
            L[i-1:i+1] = _leftorth(L[i-1],L[i])
            ry[i] = L[i-1].shape[2]

    return tt.vector.from_list(L) if isinstance(a,tt.vector) else L



def TT_factorisation_pinv(cores,n_row_modes):
    """
    assuming cores are the cores of the TT decomposition of H and n_row modes is the number
    of modes corresponding to 'prefixes' (i.e. l for H^{(2l)}), this returns the cores
    of the TT-decompositions of the pseudo-inverses of P and S, where P and S are such that
    H = PS
    """
    #print(len(cores))
    for i in range(n_row_modes):
        cores[i:i+2] = _rightorth(cores[i],cores[i+1])
    S_cores = _orthcores(cores[n_row_modes:], dir='left')
    P_cores = cores[:n_row_modes]

    c = S_cores[0]
    #print((c.reshape((c.shape[0],np.prod(c.shape[1:])))).shape)
    U,s,V = np.linalg.svd(c.reshape((c.shape[0],np.prod(c.shape[1:])),order='F'),full_matrices=False)
    P_cores[-1] = np.tensordot(P_cores[-1],U,axes=(2,0))
    S_cores[0] = (np.diag(1./s).dot(V)).reshape(c.shape,order='F')

    return P_cores,S_cores

def TT_spectral_learning(H_2l_cores, H_2l1_cores, H_l_cores):
    l = len(H_l_cores)
    P_cores, S_cores = TT_factorisation_pinv(H_2l_cores,l)

    alpha = TT_product(H_l_cores,S_cores)
    A_left = TT_product(H_2l1_cores[:l],P_cores)
    A_right = TT_product(H_2l1_cores[l+1:],S_cores)
    A = np.tensordot(A_left,H_2l1_cores[l],(2,0))
    A = np.tensordot(A,A_right,(4,0))

    omega = TT_product(H_l_cores,P_cores)

    return alpha.squeeze(), A.squeeze(), omega.squeeze()

def mps_to_tensor(mps):
    import tensorly as tl
    from tensorly import mps_to_tensor
    for i in range(len(mps)):
        mps[i] = tl.tensor(mps[i])
    ten = mps_to_tensor(mps).squeeze()
    return ten


def reg_spectral_learning(H_2l_cores, H_2l1_cores, H_l_cores):
    l = len(H_l_cores)
    rank = H_2l_cores[0].shape[-1]
    print(H_2l1_cores[0].shape)
    H_2l = mps_to_tensor(H_2l_cores)
    H_2l1 = mps_to_tensor(H_2l1_cores)
    H_l = mps_to_tensor(H_l_cores)
    #print('here')
    dim = H_l.shape[0]

    H_2l = H_2l.reshape(dim**l, dim**l)
    H_2l1 = H_2l1.reshape(dim**l, dim, dim**l)
    print(H_2l1.shape, H_2l.shape)
    H_l = H_l.reshape(-1,)
    U, D, V = np.linalg.svd(H_2l)
    P = U[:, :rank] @ np.diag(D[:rank])
    S = V[:rank, :]

    P_inv = np.linalg.pinv(P)
    S_inv = np.linalg.pinv(S)

    A = np.einsum('ij, jkl, lm->ikm', P_inv, H_2l1, S_inv)
    alpha = H_l @ S_inv
    print('here', H_l.shape, S_inv.shape, P_inv.shape)
    Omega = P_inv @ H_l

    return alpha, A, Omega




def compute_kdes_for_window_size_list(window_size_list, **option):
    kde_option = {
        'env': gym.make('Pendulum-v0'),
        'num_trajs': 100,
        'max_episode_length': 10
    }
    kde_option = {**kde_option, **option}
    for window_size in window_size_list:
        kde_address = kde_option['env'].unwrapped.spec.id + ' ' +str(window_size)
        kde_option['window_size'] = window_size
        kde = construct_KDE(**kde_option)
        f = open(kde_address, "wb")
        pickle.dump(kde, f)
        f.close()

def load_kdes_for_window_size_list(window_size_list, **option):
    kde_option = {
        'env': gym.make('Pendulum-v0'),
        'num_trajs': 100,
        'max_episode_length': 10
    }
    kde_option = {**kde_option, **option}
    kde_list = {}
    for window_size in window_size_list:
        kde_address = kde_option['env'].unwrapped.spec.id + ' ' + str(window_size)
        kde = load_kde(kde_address)
        kde_list[str(window_size)] = kde
    return kde_list

def compute_Hankels_for_window_size_list(window_size_list, **option_list):
    option_list_default = {
        'kde_option': {
            'env': gym.make('Pendulum-v0'),
            'num_trajs': 100,
            'max_episode_length': 10,
            'window_size': 5,
            'load_kde': False
        },
        'train_gen_option': {
            'env': gym.make('Pendulum-v0'),
            'num_trajs': 1000,
            'max_episode_length': 100,
            'window_size': 5},
        'validate_gen_option': {
            'env': gym.make('Pendulum-v0'),
            'num_trajs': 1000,
            'max_episode_length': 10,
            'window_size': 5},

        'hankel_option': {
            'rank': 20,
            'out_dim': 1,
            'max_length': 5,
            'device': 'cpu',
            'freeze_encoder': False,
            'mps': None,
            'init_std': 0.1
        },
        'action_encoder_option': {
            'hidden_units': [10],
            'out_dim': 10,
            'final_activation': torch.nn.Tanh(),
            'inner_activation': torch.nn.LeakyReLU(inplace=False)
        },
        'obs_encoder_option': {
            'hidden_units': [10],
            'out_dim': 10,
            'final_activation': torch.nn.Tanh(),
            'inner_activation': torch.nn.LeakyReLU(inplace=False)
        },
        'fit_option': {
            'epochs': 1000,
            'verbose': True,
            'lr': 0.001,
            'step_size': 500,
            'gamma': 0.1,
            'batch_size': 256
        }
    }
    option_list = {**option_list_default, **option_list}
    Hankels = {}
    train_encoders = True
    for window_size in window_size_list:
        for key in option_list:
            if 'window_size' in option_list[key]:
                option_list[key]['window_size'] = window_size

        kde_option = option_list['kde_option']
        train_gen_option = option_list['train_gen_option']
        validate_gen_option = option_list['validate_gen_option']
        action_encoder_option = option_list['action_encoder_option']
        obs_encoder_option = option_list['obs_encoder_option']
        hankel_option = option_list['hankel_option']
        fit_option = option_list['fit_option']

        kde = get_kde(**kde_option)

        gen_options = {
            'train_gen_option': train_gen_option,
            'validate_gen_option': validate_gen_option
        }
        train_dataset, train_loader, validate_dataset, validate_loader = get_data_loaders(kde, batch_size=fit_option['batch_size'],
                                                                                          **gen_options)
        if train_encoders:
            action_encoder_option['input_dim'] = train_dataset.action.shape[2]
            obs_encoder_option['input_dim'] = train_dataset.obs.shape[2]
            action_encoder = Encoder(**action_encoder_option)
            obs_encoder = Encoder(**obs_encoder_option)
            train_encoders = False


        hankel = Hankel(action_encoder, obs_encoder, **hankel_option)
        optimizer = optim.Adam(hankel.parameters(), lr=fit_option['lr'], amsgrad=True)
        train_lambda = lambda model: train(model, hankel_option['device'], train_loader, optimizer)
        validate_lambda = lambda model: validate(model, hankel_option['device'], validate_loader)
        fit_option['optimizer'] = optimizer
        tmp_H = fit_hankel(hankel, train_lambda, validate_lambda, **fit_option)

        Hankels[str(window_size)] = tmp_H
        action_encoder = Hankels[str(window_size_list[0])].action_encoder
        obs_encoder = Hankels[str(window_size_list[0])].obs_encoder
        hankel_option['freeze_encoder'] = True

    return Hankels

def convert_mps_tensor_to_numpy(mps):
    mps_np = []
    for core in mps:
        mps_np.append(core.cpu().detach().numpy())
    return mps_np

def merge_input_dim(mps):
    merged_mps = []
    core_shapes = []
    for core in mps:
        core_shapes.append(core.shape)
        core = np.einsum('ijkl-> lijk', core)
        core = core.reshape([core.shape[0], core.shape[1], -1])
        core = np.einsum('ijk->jki', core)
        merged_mps.append(core)
    return merged_mps, core_shapes

def unmerge_input_dim(merged_mps, core_shapes):
    mps = []
    for i, core in enumerate(merged_mps):
        core = np.einsum('jki->ijk', core)
        core = core.reshape(core_shapes[i])
        core = np.einsum('lijk->ijkl', core)
        mps.append(core)
    return mps

def run_spectral_learning(hankels):
    merged_mpss = []
    mps_shapes = []
    for key in hankels:
        hankel = hankels[key]
        # for core in hankel.mps:
        #     print(core.shape)
        merged_mps, mps_shape = merge_input_dim(convert_mps_tensor_to_numpy(hankel.mps))
        merged_mpss.append(merged_mps)
        mps_shapes.append(mps_shape)
        # for core in merged_mps:
        #     print(core.shape)

    print('here')
    alpha, A, Omega = reg_spectral_learning(merged_mpss[1], merged_mpss[0], merged_mpss[2])
    #alpha, A, Omega = TT_spectral_learning(merged_mpss[1], merged_mpss[0], merged_mpss[2])
    #print(A.shape)
    A = np.einsum('ijk->kij', A)
    A = A.reshape([A.shape[0], A.shape[1], int(np.sqrt(A.shape[2])), int(np.sqrt(A.shape[2]))])
    #print(A.shape)
    A = np.einsum('ijkl->jkli', A)
    #A = A.reshape(mps_shapes[0][1])
    #print(alpha.shape, A.shape)
    print(alpha.shape, A.shape, Omega.shape)
    return alpha, A, Omega


if __name__ =='__main__':
    length = 1
    window_size = 3
    load_hankels = True
    window_size_list = [2*length+1, 2*length, length]

    option_list = {
        'kde_option': {
            'env': gym.make('Pendulum-v0'),
            'num_trajs': 100,
            'max_episode_length': 10,
            'window_size': window_size,
            'load_kde': True
        },

        'train_gen_option': {
            'env': gym.make('Pendulum-v0'),
            'num_trajs': 1000,
            'max_episode_length': 100,
            'window_size': window_size
        },

        'validate_gen_option': {
            'env': gym.make('Pendulum-v0'),
            'num_trajs': 100,
            'max_episode_length': 100,
            'window_size': window_size
        },

        'hankel_option': {
            'rank': 20,
            'out_dim': 1,
            'window_size': window_size,
            'device': 'cpu',
            'freeze_encoder': False,
            'mps': None,
            'init_std': 0.1
        },

        'action_encoder_option': {
            'hidden_units': [10],
            'out_dim': 10,
            'final_activation': torch.nn.Tanh(),
            'inner_activation': torch.nn.LeakyReLU(inplace=False)
        },

        'obs_encoder_option': {
            'hidden_units': [10],
            'out_dim': 10,
            'final_activation': torch.nn.Tanh(),
            'inner_activation': torch.nn.LeakyReLU(inplace=False)
        },

        'fit_option': {
            'epochs': 1000,
            'verbose': True,
            'lr': 0.001,
            'step_size': 500,
            'gamma': 0.5,
            'batch_size': 256
        }
    }
    hankels_address = str(option_list['kde_option']['env'])+'_length_'+str(length)
    if load_hankels:
        f = open(hankels_address, 'rb')
        hankels = pickle.load(f)
        f.close()
    else:
        hankels = compute_Hankels_for_window_size_list(window_size_list, **option_list)
        f = open(hankels_address, 'wb')
        pickle.dump(hankels, f)
        f.close()

    alpha, A, Omega = run_spectral_learning(hankels)
    cwfa_ao = CWFA_AO(alpha, A, Omega, hankels[str(window_size_list[0])].action_encoder, hankels[str(window_size_list[0])].obs_encoder)

    #print(option_list['kde_option'])
    option_list['kde_option']['load_kde'] = True
    kde = get_kde(**option_list['kde_option'])

    gen_options = {
        'train_gen_option': option_list['train_gen_option'],
        'validate_gen_option': option_list['validate_gen_option']
    }
    train_dataset, train_loader, validate_dataset, validate_loader = get_data_loaders(kde, batch_size=256,
                                                                                      **gen_options)
    print(validate(cwfa_ao, 'cpu', train_loader))
    # print(window_size)
    # print(hankels[str(window_size)].length)
    # print(validate(hankels[str(1)], 'cpu', train_loader))
