#!/usr/bin/env python

import casscf.code.io as io
import casscf.code.Hamiltonian as Hamiltonian
import casscf.code.pyscf_tools as pyscfTools
import os
import glob
from pprint import pprint
import numpy
import pyscf
import sys
import time

from asf.wrapper import find_from_mol, sized_space_from_mol , sized_space_from_scf


system_info = dict(
                   charge=0,
                   spin=0,
                   basis="sto3g",
                   nFrozen=None, # Number of frozen spatial orbitals
                   max_memory=15000,
                   max_CASCI = 10,
                   max_CASSCF = 8,
                   max_active_space = 40,
                   ccsd = False,
                   noRDM = False,
                   nevpt2 = False,
                   nelec_active = None)

def main():
    mol = pyscfTools.genMol("O 0 0 0; H 0 0 1; H 0 1.2 0", basis=system_info["basis"], charge=system_info["charge"], spin=system_info["spin"], symmetry=False, mem=system_info["max_memory"])
    nelec = mol.nelec
    mf = pyscfTools.RHF(mol)
    mp2, natocc , natorb = pyscfTools.make_natural_orbitals(mf, True, False)
    active_space = 8
    mo_list = []
    
    pprint(mol.ao_labels())
    active_space = sized_space_from_scf(mf, 4)
    pprint(vars(active_space))
    CASCI, CIorb, CIocc, nvept = pyscfTools.CASCI(mf, active_space.nel, active_space.norb, natocc=[], natorb=active_space.mo_coeff, cas_list=active_space.mo_list, nevpt2=system_info["nevpt2"])

    pprint(vars(CASCI))
    
if __name__ == "__main__":
    main()