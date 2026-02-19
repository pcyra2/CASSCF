#!/usr/bin/env python

import casscf.code.io as io
import casscf.code.Hamiltonian as Hamiltonian
import casscf.code.pyscf_tools as pyscfTools
from casscf.code.molden_parser import load
from asf.wrapper import sized_space_from_scf
import os
import glob
from pprint import pprint
import numpy
import pyscf
import sys
import time


system_info = dict(filename="F09.xyz",
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
                   nelec_active = None,
                   ASF = False,
                   ASF_Size = 0,
                   HACK_ME = False,
                   )



def main():
    # molecule = pyscfTools.genMol(system_info["filename"], system_info["charge"], 
    #                              system_info["spin"], system_info["basis"], 
    #                              symmetry=False, mem=system_info["max_memory"])
    # nelec = molecule.nelec
    # molecule.output = "MF.out"
    # molecule.build()

    

    # hf_s = time.perf_counter()
    # mf = pyscfTools.RHF(molecule)
    # hf_e = time.perf_counter()
    # print(f"HF Energy: {mf.e_tot:.6f} Ha")

    # mf.mol.output = "MP2.out"
    # mf.mol.build()
    # mp2_s = time.perf_counter()
    # mp2, natocc , natorb = pyscfTools.make_natural_orbitals(mf, True, False)
    # mp2_e = time.perf_counter()
    # print(f"MP2 Energy: {mp2.e_tot:.6f} Ha")

    # CASCI, CIorb, CIocc, nevpt = pyscfTools.CASCI(molecule, nActiveElectrons=6, 
    #                                             nActiveOrbitals=5, natocc=natocc, 
    #                                             natorb=natorb, cas_list=[115, 116, 121, 122, 128], 
    #                                             max_run=system_info["max_CASCI"], nevpt2 = system_info["nevpt2"])
    # print(f"CASCI Energy: {CASCI.e_tot:.6f} Ha")

    # print("Testing molden reader")
    # molecule, mo_energies, natorb , natocc, _,_ = load("MP2.molden", verbose=1)
    # CASCI, CIorb, CIocc, nevpt = pyscfTools.CASCI(molecule, nActiveElectrons=6, 
    #                                             nActiveOrbitals=5, natocc=natocc, 
    #                                             natorb=natorb, cas_list=[115, 116, 121, 122, 128], 
    #                                             max_run=system_info["max_CASCI"], nevpt2 = system_info["nevpt2"])
    # print(f"CASCI Energy: {CASCI.e_tot:.6f} Ha")

    print("Reading in orca molden")


    molecule, mo_energies, natorb , natocc, _,_ = load("orca_mp2.molden", verbose=1)
    molecule.charge=0
    molecule.spin = 0
    molecule.unit="Angstrom"
    # molecule.verbose=6
    molecule.symmetry = False
    molecule.build()

    # pyscf.tools.molden.from_mo(molecule, "orca_2.molden", natorb, occ=natocc)

    # molecule, mo_energies, natorb , natocc, _,_ = load("orca_2.molden", verbose=1)
    # molecule.output = "orca.out"
    # molecule.build()
    # pprint(vars(molecule))

    CASCI, CIorb, CIocc, nevpt = pyscfTools.CASCI(molecule, nActiveElectrons=6, 
                                                nActiveOrbitals=5, natocc=natocc, 
                                                natorb=natorb, cas_list=[118, 119, 120, 121, 122], 
                                                max_run=system_info["max_CASCI"], nevpt2 = system_info["nevpt2"])
    print(f"CASCI Energy: {CASCI.e_tot:.6f} Ha")

if __name__ == "__main__":
    main()