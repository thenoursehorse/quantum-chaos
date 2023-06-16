import time
import numpy as np

import os
import argparse
import textwrap

from kicked_boson.quantum.dicke import Dicke

if __name__ == '__main__':
    description = textwrap.dedent('''\
         ''')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)
    parser.add_argument('-N', type=int, default=32) #16
    parser.add_argument('-Nc', type=int, default=320) #320
    parser.add_argument('-kappa', type=float, default=1.2)
    parser.add_argument('-lambda0', type=float, default=0.5)
    parser.add_argument('-root_folder', type=str, default='./')
    args = parser.parse_args()
    print(vars(args))

    def run(save=False, show=True):
        folder = f'{args.root_folder}/figs/Dicke/N{args.N}_Nc{args.Nc}/kappa{args.kappa:.2f}_lambda{args.lambda0:.2f}'
        if save:
            os.makedirs(f'{folder}', exist_ok=True)

        start = time.time()
        model = Dicke(N=args.N,
                      Nc=args.Nc,
                      num_ensembles=1,
                      kappa=args.kappa,
                      lambda0=args.lambda0,
                      folder=folder) 
        end = time.time()
        print("Unitary construction took", end-start)
        
        start = time.time()
        model.set_spectral_functions(window=0)
        end = time.time()
        print("Spectral function construction took", end-start)
        
        #start = time.time()
        #model.set_unitary_evolve(num_ensembles=None)
        #end = time.time()
        #print("Time evolve unitaries took", end-start)
        
        #start = time.time()
        #model.set_unitary_fidelity()
        #end = time.time()
        #print("Unitaries fidelity took", end-start)

        start = time.time()
        model.plot_eigenenergies(save=save, show=show)
        model.plot_ratios(save=save, show=show)
        model.plot_frame_potential(estimate=False, save=save, show=show, window=0)
        model.plot_spectral_functions(save=save, show=show)
        model.plot_loschmidt_echo(save=save, show=show)
        model.unfold_energies(save=save, show=show, plot=True)
        model.plot_spacings(save=save, show=show)
        end = time.time()
        print("Plotting took", end-start)
        
        return model

    model = run()
    r_avg, r_err = model.average_level_ratios()
    p_pois, p_goe, p_gue = model.chi_distance()