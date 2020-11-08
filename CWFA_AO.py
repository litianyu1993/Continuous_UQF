import numpy as np
import tensorly as tl
class CWFA_AO:

    def __init__(self, alpha, A, Omega):
        self.alpha = alpha
        self.A = A
        self.Omega = Omega
        self.num_states = alpha.shape[0]
        self.input_dim = A.shape[1]
        self.output_dim = Omega.shape[1] if Omega.ndim > 1 else 1


    def update_dynamics(self, prev, action, obs):

        tmp = tl.tenalg.contract(self.A, 1, action, 0)
        tmp = tl.tenalg.contract(obs, 0, tmp, 1)
        next = tl.tenalg.contract(tmp, 0, prev, 0)
        next = next.reshape(-1, )
        return next

    def term_dynamics(self, prev):
        term = np.tensordot(prev, self.Omega, [prev.ndim - 1, 0])
        return term

    def predict(self, act_seq, obs_seq):
        current_state = self.alpha
        for a, o in zip(act_seq, obs_seq):
            current_state = self.update_dynamics(current_state, a, o)
        term = self.term_dynamics(current_state).ravel()
        pred = term if self.output_dim > 1 else term[0]
        return np.asarray(pred)


    def build_true_Hankel_tensor(self,l):
        H = self.alpha
        for i in range(l):
            H = np.tensordot(H,self.A,[H.ndim-1,0])
        H = np.tensordot(H,self.Omega,[H.ndim-1,0])
        return H


if __name__ == '__main__':
    rank = 5
    action_dim = 3
    obs_dim = 4
    out_dim = 2
    alpha = np.random.normal(0, 1, [rank])
    Omega = np.random.normal(0, 1, [rank, out_dim])
    A = np.random.normal(0, 1, [rank, action_dim, obs_dim, rank])
    cwfa = CWFA_AO(alpha, A, Omega)

    action_seqs = np.random.normal(0, 1, [100, 5, action_dim])
    obs_seqs = np.random.normal(0, 1, [100, 5, obs_dim])
    for i in range(len(action_seqs)):
        print(action_seqs[i].shape, obs_seqs[i].shape)
        print(cwfa.predict(action_seqs[i], obs_seqs[i]))