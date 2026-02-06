from setuptools import setup
# set up using "pip install -e ."
setup(
    name='casscf',
    version='1.0',
    py_modules=['casscf'],
    entry_points={
        'console_scripts': [
             "CASSCF = casscf.experiments.CASSCF:main",
        ],
    },
    install_requires=["pyscf", "qiskit", "qiskit-nature", 
                      "openfermion", "ase", "h5py"
                        ]
)
