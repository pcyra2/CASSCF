from ase.atoms import Atoms
from  pyscf import gto, scf, ao2mo
from pyscf.lo.boys import Boys
from pyscf.lo.edmiston import Edmiston
from pyscf.lo.pipek import PipekMezey
import casscf.code.pyscf_tools as pyscf_tools
# from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
import numpy as np


def construct_spin_one_body(blocks, mode_count):
    assert len(blocks) == 2

    zero_block = np.zeros((mode_count, mode_count))

    return np.block([[blocks[0], zero_block], [zero_block, blocks[1]]])


def construct_spin_two_body(blocks, mode_count):
    assert len(blocks) == 4

    zero_block = np.zeros((mode_count, mode_count, mode_count, mode_count))
    b = np.zeros((2, 2, 2, 2)).tolist()

    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    b[i][j][k][l] = zero_block

    b[0][0][0][0] = blocks[0]
    b[0][0][1][1] = blocks[1]
    b[1][1][0][0] = blocks[2]
    b[1][1][1][1] = blocks[3]

    return np.block(b) * (-1 / 2)


# def build_molecular_integrals(mol: gto.mole.Mole, UHF=False,):
def build_molecular_integrals(mol: gto.mole.Mole, hf: scf.uhf.UHF|scf.rhf.RHF, UHF:bool, ):
    # Set up output
    # orbitals = {k: {} for k in ["boys", "molecular", "atomic"]}
    orbitals = {k: {} for k in ["molecular", "atomic", "boys", "EdmistonRuednberg", "PipekMezey"]}

    spin = np.array([0, 1])

    # quick branch for UHF vs RHF
    if UHF == True:
        # hf = scf.UHF(mol)
        # hf.kernel()
        mf_coeff = (
            hf.mo_coeff
        )  # mean-field coefficients of molecualr orbitals (in this case hf)
        mf_occ = hf.mo_occ
    else:
        # hf = scf.RHF(mol)
        # hf.kernel()
        mf_coeff = np.array([hf.mo_coeff, hf.mo_coeff])
        mf_occ = [hf.mo_occ, hf.mo_occ]
    # print(f"{mf_occ.shape=}")
    ##### Freeze core
    # if nFrozen >0:
    #     frozen = int(nFrozen/2)
    #     # mf_coeff = np.array([mf_coeff[0][:][frozen:],mf_coeff[1][:][frozen:]])
    #     mf_occ = np.array([mf_occ[0][frozen:], mf_occ[1][frozen:]])
    
    # print(f"{mf_occ.shape=}")
    lmo_coeff = {}
    occupy = hf.get_occ()

    # Extract atomic integrals
    s = mol.intor("int1e_ovlp")
    # if nFrozen > 0:
    #     s = np.array([s[i][frozen:] for i in range(frozen, len(s))])
    eigval, eigvec = np.linalg.eigh(s)
    s_inv_sqrt = np.dot(eigvec, np.dot(np.diag(eigval**-0.5), eigvec.T))
    ao_coeff = s_inv_sqrt
    print("initiating integrals")
    one_body_atomic = mol.intor("int1e_kin") + mol.intor("int1e_nuc") 
    two_body_atomic = mol.intor("int2e")
    mode_count = len(one_body_atomic)
    print("atomic_init")
    one_body_atomic_blocks = [
            np.transpose(ao_coeff) @ one_body_atomic @ ao_coeff,
            np.transpose(ao_coeff) @ one_body_atomic @ ao_coeff
    ]

    two_body_atomic_blocks = [
        ao2mo.general(
            mol,
            (ao_coeff, ao_coeff, ao_coeff, ao_coeff),
            aosym="s1",
            compact=False,
        ).reshape(mode_count, mode_count, mode_count, mode_count),
        ao2mo.general(
            mol,
            (ao_coeff, ao_coeff, ao_coeff, ao_coeff),
            aosym="s1",
            compact=False,
        ).reshape(mode_count, mode_count, mode_count, mode_count),
        ao2mo.general(
            mol,
            (ao_coeff, ao_coeff, ao_coeff, ao_coeff),
            aosym="s1",
            compact=False,
        ).reshape(mode_count, mode_count, mode_count, mode_count),
        ao2mo.general(
            mol,
            (ao_coeff, ao_coeff, ao_coeff, ao_coeff),
            aosym="s1",
            compact=False,
        ).reshape(mode_count, mode_count, mode_count, mode_count),
    ]

    print("Atomic blocked.")

    orbitals["atomic"]["one_body"] = construct_spin_one_body(
        one_body_atomic_blocks, mode_count
    )
    orbitals["atomic"]["two_body"] = construct_spin_two_body(
        two_body_atomic_blocks, mode_count
    )

    # Compute molecular integrals
    print("Computing molecular")
    one_body_molecular_blocks = [
        np.transpose(mf_coeff[0]) @ one_body_atomic @ mf_coeff[0],
        np.transpose(mf_coeff[1]) @ one_body_atomic @ mf_coeff[1],
    ]

    two_body_molecular_blocks = [
        ao2mo.general(
            mol,
            (mf_coeff[0], mf_coeff[0], mf_coeff[0], mf_coeff[0]),
            aosym="s1",
            compact=False,
        ).reshape(mode_count, mode_count, mode_count, mode_count),
        ao2mo.general(
            mol,
            (mf_coeff[0], mf_coeff[0], mf_coeff[1], mf_coeff[1]),
            aosym="s1",
            compact=False,
        ).reshape(mode_count, mode_count, mode_count, mode_count),
        ao2mo.general(
            mol,
            (mf_coeff[1], mf_coeff[1], mf_coeff[0], mf_coeff[0]),
            aosym="s1",
            compact=False,
        ).reshape(mode_count, mode_count, mode_count, mode_count),
        ao2mo.general(
            mol,
            (mf_coeff[1], mf_coeff[1], mf_coeff[1], mf_coeff[1]),
            aosym="s1",
            compact=False,
        ).reshape(mode_count, mode_count, mode_count, mode_count),
    ]
    print("Molecular computed. ")

    orbitals["molecular"]["one_body"] = construct_spin_one_body(
        one_body_molecular_blocks, mode_count
    )
    orbitals["molecular"]["two_body"] = construct_spin_two_body(
        two_body_molecular_blocks, mode_count
    )
    orbitals["molecular"]["alpha"] = one_body_molecular_blocks[0]
    orbitals["molecular"]["beta"] = one_body_molecular_blocks[1]
    orbitals["molecular"]["alpha_alpha"] = two_body_molecular_blocks[0]
    orbitals["molecular"]["beta_beta"] = two_body_molecular_blocks[3]
    orbitals["molecular"]["alpha_beta"] = two_body_molecular_blocks[1] 
    
    # # Compute Boys integrals

    # for sp in spin:
    #     # localisation from https://github.com/pyscf/pyscf/issues/820
    #     # Localise
    #     loc = Boys(mol, mf_coeff[sp][:, mf_occ[sp] >= 0])
    #     loc.verbose = 9
    #     loc.init_guess = None
    #     # get localised MO coeffiecients
    #     lmo_coeff[
    #         sp
    #     ] = (
    #         loc.kernel()
    #     )

    # one_body_boys_blocks = [
    #     np.transpose(lmo_coeff[0]) @ one_body_atomic @ lmo_coeff[0],
    #     np.transpose(lmo_coeff[1]) @ one_body_atomic @ lmo_coeff[1],
    # ]
    # two_body_boys_blocks = [
    #     ao2mo.general(
    #         mol,
    #         (lmo_coeff[0], lmo_coeff[0], lmo_coeff[0], lmo_coeff[0]),
    #         aosym="s1",
    #         compact=False,
    #     ).reshape(mode_count, mode_count, mode_count, mode_count),
    #     ao2mo.general(
    #         mol,
    #         (lmo_coeff[0], lmo_coeff[0], lmo_coeff[1], lmo_coeff[1]),
    #         aosym="s1",
    #         compact=False,
    #     ).reshape(mode_count, mode_count, mode_count, mode_count),
    #     ao2mo.general(
    #         mol,
    #         (lmo_coeff[1], lmo_coeff[1], lmo_coeff[0], lmo_coeff[0]),
    #         aosym="s1",
    #         compact=False,
    #     ).reshape(mode_count, mode_count, mode_count, mode_count),
    #     ao2mo.general(
    #         mol,
    #         (lmo_coeff[1], lmo_coeff[1], lmo_coeff[1], lmo_coeff[1]),
    #         aosym="s1",
    #         compact=False,
    #     ).reshape(mode_count, mode_count, mode_count, mode_count),
    # ]

    # orbitals["boys"]["one_body"] = construct_spin_one_body(
    #     one_body_boys_blocks, mode_count
    # )
    # orbitals["boys"]["two_body"] = construct_spin_two_body(
    #     two_body_boys_blocks, mode_count
    # )


    # ERmo_coeff = {}
    # for sp in spin:
    #     # localisation from https://github.com/pyscf/pyscf/issues/820
    #     # Localise
    #     loc = Edmiston(mol, mf_coeff[sp][:, mf_occ[sp] >= 0])
    #     loc.verbose = 9
    #     loc.init_guess = None
    #     # get localised MO coeffiecients
    #     ERmo_coeff[
    #         sp
    #     ] = (
    #         loc.kernel()
    #     )
    # one_body_boys_blocks = [
    #     np.transpose(ERmo_coeff[0]) @ one_body_atomic @ ERmo_coeff[0],
    #     np.transpose(ERmo_coeff[1]) @ one_body_atomic @ ERmo_coeff[1],
    # ]
    # two_body_boys_blocks = [
    #     ao2mo.general(
    #         mol,
    #         (ERmo_coeff[0], ERmo_coeff[0], ERmo_coeff[0], ERmo_coeff[0]),
    #         aosym="s1",
    #         compact=False,
    #     ).reshape(mode_count, mode_count, mode_count, mode_count),
    #     ao2mo.general(
    #         mol,
    #         (ERmo_coeff[0], ERmo_coeff[0], ERmo_coeff[1], ERmo_coeff[1]),
    #         aosym="s1",
    #         compact=False,
    #     ).reshape(mode_count, mode_count, mode_count, mode_count),
    #     ao2mo.general(
    #         mol,
    #         (ERmo_coeff[1], ERmo_coeff[1], ERmo_coeff[0], ERmo_coeff[0]),
    #         aosym="s1",
    #         compact=False,
    #     ).reshape(mode_count, mode_count, mode_count, mode_count),
    #     ao2mo.general(
    #         mol,
    #         (ERmo_coeff[1], ERmo_coeff[1], ERmo_coeff[1], ERmo_coeff[1]),
    #         aosym="s1",
    #         compact=False,
    #     ).reshape(mode_count, mode_count, mode_count, mode_count),
    # ]

    # orbitals["EdmistonRuednberg"]["one_body"] = construct_spin_one_body(
    #     one_body_boys_blocks, mode_count
    # )
    # orbitals["EdmistonRuednberg"]["two_body"] = construct_spin_two_body(
    #     two_body_boys_blocks, mode_count
    # )
    orbitals["EdmistonRuednberg"]["one_body"], orbitals["EdmistonRuednberg"]["two_body"] = FunkyOrbitals("EdmistonRuednberg", hf, mol, one_body_atomic)
    orbitals["boys"]["one_body"], orbitals["boys"]["two_body"] = FunkyOrbitals("boys", hf, mol, one_body_atomic)
    orbitals["PipekMezey"]["one_body"], orbitals["PipekMezey"]["two_body"] = FunkyOrbitals("PipekMezey", hf, mol, one_body_atomic)
    return (orbitals, occupy)

def FunkyOrbitals(orbType:str, mf, mol, one_body_atomic, ):
    command = {"boys":Boys,
               "PipekMezey": PipekMezey,
               "EdmistonRuednberg": Edmiston,
               }
    spin = np.array([0, 1])
    lmo_coeff = {}
    mode_count = len(one_body_atomic)
    mf_coeff = (
            mf.mo_coeff
        )  # mean-field coefficients of molecualr orbitals (in this case hf)
    mf_occ = mf.mo_occ
    for sp in spin:
        # localisation from https://github.com/pyscf/pyscf/issues/820
        # Localise
        loc = command[orbType](mol, mf_coeff[sp][:, mf_occ[sp] >= 0])
        loc.verbose = 9
        loc.init_guess = None
        # get localised MO coeffiecients
        lmo_coeff[
            sp
        ] = (
            loc.kernel()
        )
    one_body_blocks = [
        np.transpose(lmo_coeff[0]) @ one_body_atomic @ lmo_coeff[0],
        np.transpose(lmo_coeff[1]) @ one_body_atomic @ lmo_coeff[1],
    ]
    two_body_blocks = [
        ao2mo.general(
            mol,
            (lmo_coeff[0], lmo_coeff[0], lmo_coeff[0], lmo_coeff[0],),
            aosym="s1",
            compact=False,
        ).reshape(mode_count, mode_count, mode_count, mode_count),
        ao2mo.general(
            mol,
            (lmo_coeff[0], lmo_coeff[0], lmo_coeff[1], lmo_coeff[1]),
            aosym="s1",
            compact=False,
        ).reshape(mode_count, mode_count, mode_count, mode_count),
        ao2mo.general(
            mol,
            (lmo_coeff[1], lmo_coeff[1], lmo_coeff[0], lmo_coeff[0]),
            aosym="s1",
            compact=False,
        ).reshape(mode_count, mode_count, mode_count, mode_count),
        ao2mo.general(
            mol,
            (lmo_coeff[1], lmo_coeff[1], lmo_coeff[1], lmo_coeff[1]),
            aosym="s1",
            compact=False,
        ).reshape(mode_count, mode_count, mode_count, mode_count),
    ]
    return construct_spin_one_body(one_body_blocks, mode_count), construct_spin_two_body(two_body_blocks, mode_count)
    

# def extract_integral_tensors(
#     atoms: Atoms, UHF: bool, basis: str
# ) -> (np.ndarray, np.ndarray):
#     # Build integrals
#     integrals, occupancy = build_molecular_integrals(
#         atoms, UHF=UHF, 
#     )
def extract_integral_tensors(
    molecule,MF, basis: str, UHF:bool, 
) -> (np.ndarray, np.ndarray):
    # Build integrals
    integrals, occupancy = build_molecular_integrals(
        # atoms, UHF=UHF, 
        molecule, MF, UHF
    )
    assert basis in ["atomic", "molecular", "boys"]

    # Extract appropriate tensors
    one_body = integrals[basis]["one_body"]
    two_body = integrals[basis]["two_body"]

    # Chemist's -> Physicist's notation
    two_body = np.swapaxes(two_body, 1, 2)

    return (one_body, two_body,integrals)

def  transform_eri_to_orthogonal_basis(eri_ao, S_inv_sqrt):
    nbf = S_inv_sqrt.shape[0]
    eri_ao = eri_ao.reshape(nbf, nbf, nbf, nbf)
    
    eri_ao = np.einsum('pqrs,pi->iqrs', eri_ao, S_inv_sqrt)
    eri_ao = np.einsum('iqrs,qj->ijrs', eri_ao, S_inv_sqrt)
    eri_ao = np.einsum('ijrs,rk->ijks', eri_ao, S_inv_sqrt)
    eri_ao = np.einsum('ijks,sl->ijkl', eri_ao, S_inv_sqrt)
    
    return eri_ao

def build_molecular_integrals_uhf(mf, frozen=0, mol=None):
    integrals = {}
    if not mol:
        mol = mf.mol

    mo_coeff = mf.mo_coeff
    ncore = frozen
    ncas = mol.nao - frozen

    mo_core = [mo[:, :ncore] for mo in mo_coeff]
    mo_cas = [mo[:, ncore:ncore + ncas] for mo in mo_coeff]

    hcore = mf.get_hcore()
    energy_core = mol.energy_nuc() + sum(np.einsum('ii->', mo.T @ hcore @ mo) for mo in mo_core)
    print("Starting")
    eri_full_mo = [
        ao2mo.general(mol, (mo_coeff[0],) * 4, compact=False, ioblk_size=100, verbose=7).reshape((mo_coeff[0].shape[1],) * 4),
        ao2mo.general(mol, (mo_coeff[0], mo_coeff[0], mo_coeff[1], mo_coeff[1]), compact=False, ioblk_size=100, verbose=7).reshape(
            (mo_coeff[0].shape[1], mo_coeff[0].shape[1], mo_coeff[1].shape[1], mo_coeff[1].shape[1])),
        ao2mo.general(mol, (mo_coeff[1], mo_coeff[1], mo_coeff[0], mo_coeff[0]), compact=False, ioblk_size=100, verbose=7).reshape(
            (mo_coeff[1].shape[1], mo_coeff[1].shape[1], mo_coeff[0].shape[1], mo_coeff[0].shape[1])),
        ao2mo.general(mol, (mo_coeff[1],) * 4, compact=False, ioblk_size=100, verbose=7).reshape((mo_coeff[1].shape[1],) * 4)
    ]
    print("Finished atomics")
    h1_mo = [mo.T @ hcore @ mo for mo in mo_coeff]
    one_body_molecular_blocks = []

    # eri_full_mo is aaaa, aabb, bbaa, bbbb

    for spin in range(2):
        h1_active = h1_mo[spin][ncore:ncore + ncas, ncore:ncore + ncas].copy()
        h1_active += np.einsum('uvii->uv',
                               eri_full_mo[spin * 3][ncore:ncore + ncas, ncore:ncore + ncas, :ncore, :ncore])
        h1_active -= np.einsum('uiiv->uv',
                               eri_full_mo[spin * 3][ncore:ncore + ncas, :ncore, :ncore, ncore:ncore + ncas])
        h1_active += np.einsum('uvii->uv',
                               eri_full_mo[1 if spin == 0 else 2][ncore:ncore + ncas, ncore:ncore + ncas, :ncore,
                               :ncore])
        one_body_molecular_blocks.append(h1_active)

    energy_core += 0.5 * sum(np.einsum('iijj->', eri[:ncore, :ncore, :ncore, :ncore]) -
                             np.einsum('ijji->', eri[:ncore, :ncore, :ncore, :ncore])
                             for eri in [eri_full_mo[0], eri_full_mo[3]])
    energy_core += 0.5 * np.einsum('iijj->', eri_full_mo[1][:ncore, :ncore, :ncore, :ncore])
    energy_core += 0.5 * np.einsum('iijj->', eri_full_mo[2][:ncore, :ncore, :ncore, :ncore])

    print("Starting molecular")
    two_body_molecular_blocks = [
        ao2mo.general(mol, (mo_cas[0],) * 4, compact=False).reshape((ncas,) * 4),
        ao2mo.general(mol, (mo_cas[0], mo_cas[0], mo_cas[1], mo_cas[1]), compact=False).reshape((ncas,) * 4),
        ao2mo.general(mol, (mo_cas[1], mo_cas[1], mo_cas[0], mo_cas[0]), compact=False).reshape((ncas,) * 4),
        ao2mo.general(mol, (mo_cas[1],) * 4, compact=False).reshape((ncas,) * 4)
    ]
    print("Finished molecular")
    mode_count = ncas
    print(mode_count)
    integrals["one_body"] = construct_spin_one_body(one_body_molecular_blocks, mode_count)
    integrals["two_body"] = construct_spin_two_body(two_body_molecular_blocks, mode_count)

    orbitals = {"molecular":{}}
    orbitals["molecular"]["one_body"] = integrals["one_body"]
    orbitals["molecular"]["two_body"] = integrals["two_body"]
    orbitals["molecular"]["alpha"] = one_body_molecular_blocks[0]
    orbitals["molecular"]["beta"] = one_body_molecular_blocks[1]
    orbitals["molecular"]["alpha_alpha"] = two_body_molecular_blocks[0]
    orbitals["molecular"]["beta_beta"] = two_body_molecular_blocks[3]
    orbitals["molecular"]["alpha_beta"] = two_body_molecular_blocks[1] 

    return integrals["one_body"], np.swapaxes(integrals["two_body"], 1, 2), energy_core, orbitals