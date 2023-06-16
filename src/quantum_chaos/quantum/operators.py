
import qutip as qt

def get_destroy_operators(dims, excitations=1):
    return qt.enr_destroy(dims, excitations)

def get_sigma_ops(N, axis):
    import qutip as qt
    si = qt.qeye(2)
   
    if axis == 'x':
        s = qt.sigmax()
    elif axis == 'y':
        s = qt.sigmay()
    elif axis == 'z':
        s = qt.sigmaz()
    elif axis == '+':
        s = qt.sigmap()
    elif axis == '-':
        s = qt.sigmam()
    else:
        raise ValueError('must be x,y,z,+,- for spin operators')

    s_list = []
    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = s
        s_list.append(qt.tensor(op_list))
    return s_list        