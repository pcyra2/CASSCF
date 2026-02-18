import pyscf
import pyscf.fci as fci
from pyscf import cc, mcscf, mp, mrpt
from pyscf.mp.dfmp2_native import DFRMP2
from pyscf.mp.dfump2_native import DFUMP2
import pyscf.tools.cubegen as cubegen
from pyscf.mcscf import avas


import os
import numpy
import scipy
from pprint import pprint



def genMol(atoms: str|list, charge: int, spin:int, basis: str, symmetry:bool, mem:int)->pyscf.M:
    """Generates a pySCF molecule from a .xyz file

    Args:
        path (str|list): Path to file, including extension or list of atoms. This list can either be "atom x y z" or [atom, x, y, z]
        charge (int): Net charge
        spin (int): Net spin in pySCF format (2S)
        basis (str): Basis set to generate the molecule in 
        symmetry (bool): Whether to use symmetry

    Returns:
        mol (pyscf.M): pySCF molecule object. 
    """
    print(atoms)
    if type(atoms) == str:
        if ".xyz" in atoms:
            assert os.path.isfile(atoms), "Coordinate file does not exist."
            mol = pyscf.gto.Mole(atom=atoms, unit="Ang")
        elif ";" in atoms:
            mol = pyscf.gto.Mole(atom=atoms, unit="Ang")
        else:
            raise Exception("Unknown atoms")
    elif type(atoms) == list:
        text = ""
        for atom in atoms:
            if len(atom) == 4:
                text = text + "; "+ str(atom[0]) + " "  + str(atom[1]) + " " + str(atom[2]) + " " + str(atom[3]) 
            else:
                text = text + "; " + atom
        mol = pyscf.gto.Mole(atom=text)
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.symmetry = symmetry
    mol.verbose = 6
    mol.output="./mol.out"
    mol.max_memory = mem
    # mol.unit="Ang"
    mol.build()
    return mol

def UHF(mol: pyscf.M)->pyscf.scf.UHF:
    """performs UHF on the molecule

    Args:
        mol (pyscf.M): pySCF Molecule object

    Returns:
        HF (pyscf.scf.UHF): a completed pySCF UHF object
    """
    HF = pyscf.scf.UHF(mol)
    HF.max_cycle = 300
    HF.kernel()
    return HF

def RHF(mol: pyscf.M)->pyscf.scf.RHF:
    """performs RHF on the molecule

    Args:
        mol (pyscf.M): pySCF Molecule object

    Returns:
        HF (pyscf.scf.UHF): a completed pySCF RHF object
    """
    HF = pyscf.scf.RHF(mol)
    HF.kernel()
    return HF

def FCI(HF: pyscf.scf.uhf.UHF, UHF: bool, )-> float:
    """Performs FullCI on a pySCF mean field object

    Args:
        HF (pyscf.scf.UHF or pyscf.scf.RHF): completed pySCF meanfield object

    Returns:
        Energy (float): Energy of the system in Hatree
        rdm1 (list): Reduced 1-body density matrix
        rdm2 (list): Reduces 2-body density matrix
    """
    norb = len(HF.mo_coeff[0])*2
    cisolver = fci.FCI(HF)
    Energy, fcivec = cisolver.kernel()
    occ = HF.mo_occ
    if UHF == True:
        alpha = int(numpy.sum(occ[0]))
        beta = int(numpy.sum(occ[1]))
    else: 
        alpha = int(numpy.sum(occ[0])/2)
        beta = alpha
    print(f"{alpha=}, {beta=}")
    print(f"{norb=}")
    # rdm1, rdm2 = cisolver.make_rdm12(fcivec, int(norb), (alpha, beta)) # Currently not working correctly. #TODO: Fix this
    rdm1 = False
    rdm2 = False
    return Energy, rdm1, rdm2

def CCSD(MF, FrozenCore:bool, Tripples:bool):
    """
    Perform a coupled cluster singles and doubles (CCSD) calculation using PySCF.

    Parameters:
        MF (pyscf.scf.RHF): A Hartree-Fock object initialized with the molecular 
                            information. This is typically obtained from pyscf.
        FrozenCore (bool): If True, include frozen core approximation in the CCSD calculation.
        Tripples (bool): If True, include triples corrections to the CCSD calculation.

    Returns:
        tuple: A tuple containing four elements:
            - myCC (pyscf.cc.CCSD): The coupled cluster object after running the calculation.
            - et (float or None): Energy of the coupled cluster calculation  triple correction, or None if Tripples is False.
            - rdm1 (numpy.ndarray): One-particle reduced density matrix from the CCSD calculation.
            - rdm2 (numpy.ndarray): Two-particle reduced density matrix from the CCSD calculation.
    """
    myCC = cc.CCSD(MF)
    myCC.max_cycle=300
    if FrozenCore == True:
        myCC.set_frozen()
    myCC.run()
    if Tripples == True:
        et = myCC.ccsd_t()
    else: 
        et = 0
    # rdm1 = myCC.make_rdm1()
    # rdm2 = myCC.make_rdm2()
    return myCC, et

def MP2(mf, restricted:bool, density_fitting:bool, frozen_core:bool|int):
    """
    Perform a second-order Møller-Plesset perturbation theory (MP2) calculation using PySCF.

    Parameters:
        mf (pyscf.scf.RHF or pyscf.scf.UHF): The mean-field object representing the molecular system.
        restricted (bool): If True, perform a restricted MP2 calculation; otherwise, perform an unrestricted MP2 calculation.
        density_fitting (bool): If True, use density fitting to speed up the MP2 calculation.
    """
    if density_fitting:

        if restricted:
            if frozen_core:
                mp2 = DFRMP2(mf, frozen=frozen_core)
            else:
                mp2 = DFRMP2(mf)
        else:
            if frozen_core:
                mp2 = DFUMP2(mf, frozen=frozen_core)
            else:
                mp2 = DFUMP2(mf)
    else:
        mp2 = mp.MP2(mf)
        if frozen_core:
            mp2.set_frozen()
    mp2.kernel()
    return mp2

def make_natorbs_mp2(mp2, rdm1_mo=None, relaxed=False):
        '''
        Calculate natural orbitals.
        Note: the most occupied orbitals come first (left)
              and the least occupied orbitals last (right).

        Args:
            rdm1_mo : 1-RDM in MO basis
                      the function calculates a density matrix if none is provided
            relaxed : calculated relaxed or unrelaxed density matrix

        Returns:
            natural occupation numbers, natural orbitals
        '''
        if rdm1_mo is None:
            dm = mp2.make_rdm1()
        elif isinstance(rdm1_mo, numpy.ndarray):
            dm = rdm1_mo
        else:
            raise TypeError('rdm1_mo must be a 2-D array')

        eigval, eigvec = numpy.linalg.eigh(dm)
        natocc = numpy.flip(eigval)
        natorb = numpy.dot(mp2.mo_coeff, numpy.fliplr(eigvec))
        return natocc, natorb

def make_natural_orbitals(mf, FrozenCore: bool, DensityFit: bool):
    if type(mf) == pyscf.scf.rhf.RHF:
        restricted = True
    else:
        restricted = False
    mp2 = MP2(mf, restricted=restricted,density_fitting=DensityFit, frozen_core=FrozenCore)
    if DensityFit == False:
        natocc, natorb = make_natorbs_mp2(mp2)
    else:
        natocc, natorb = mp2.make_natorbs()
    return mp2, natocc, natorb

def CASSCF(mf, nActiveElectrons:int, nActiveOrbitals:int, natocc:numpy.array = None, natorb:numpy.array = None, 
           NFrozen:int = 0, cas_list = None, max_run :int = 16,):
    nevpt = None
    if natocc is None or natorb is None:
        if NFrozen > 0:
            FrozenCore = True
        else:
            FrozenCore = False
        _, natocc, natorb = make_natural_orbitals(mf, FrozenCore=FrozenCore, DensityFit=False)

    cas = mcscf.CASSCF(mf, nActiveOrbitals, nActiveElectrons)
    if cas_list is not None:
        natorb = cas.sort_mo(cas_list, natorb)
    # cas.output = f"ActiveSpace_{nActiveElectrons}.out"
    cas.natorb = True
    cas.max_cycle_micro = 10
    cas.internal_rotation = True
    if NFrozen > 0:
        cas.frozen = NFrozen
    cas.conv_tol_grad = 1e-3
    cas.conv_tol = 1e-8
    cas.with_dep4 = True
    if nActiveElectrons <max_run:
        cas.kernel(natorb)
        # natorb, natocc = make_natorbs_mp2(cas)
        natorb = cas.cas_natorb()[0]
        natocc = cas.cas_natorb()[2]

    return cas, natorb, natocc, nevpt

def CASCI(mf, nActiveElectrons:int, nActiveOrbitals:int, natocc:numpy.array = None, natorb:numpy.array = None,  cas_list=None, max_run:int = 16, nevpt2:bool=False):
    nevpt = None
    if natocc is None or natorb is None:
        _, natocc, natorb = make_natural_orbitals(mf, FrozenCore=False, DensityFit=False)

    cas = mcscf.CASCI(mf, nActiveOrbitals, nActiveElectrons)
    if cas_list is not None:
        natorb = cas.sort_mo(cas_list, natorb)
    # cas.output = f"ActiveSpace_{nActiveElectrons}.out"
    cas.natorb = True
    # cas.internal_rotation = True

    if nActiveElectrons < max_run:
        cas.kernel(natorb)
        if nevpt2:
            nevpt = mrpt.nevpt2.NEVPT(cas)
            nevpt.kernel()
            # pprint(vars(nevpt))
    return cas, natorb, natocc, nevpt

def genCube(coeffs:numpy.array, Molecule: pyscf.M,  path:str, UHF:bool = False, basis:str = "molecular") -> None: # pragma: no cover
    """Generates cube files
    """
    if UHF == False:
        for i in  range(len(coeffs[:,0])):
            # print(mf.mo_coeff)
            try:
                os.mkdir(f"{path}/Orbitals")
            except FileExistsError:
                pass
            if os.path.isfile( f"{path}Orbitals/Orbital_{i}-0.cube") == False:
                cubegen.orbital(Molecule, f"{path}Orbitals/Orbital_{i}-0.cube", coeffs[:,i])
    else:
        for i in range(len(coeffs[0][:])):
            try:
                os.mkdir(f"{path}/Orbitals")
            except FileExistsError:
                pass
            if os.path.isfile(f"{path}Orbitals/Orbital_{i}-0.cube") == False:
                cubegen.orbital(Molecule, f"{path}Orbitals/Orbital_{i}-0.cube", coeffs[0][i])
                cubegen.orbital(Molecule, f"{path}Orbitals/Orbital_{i}-1.cube", coeffs[1][i])