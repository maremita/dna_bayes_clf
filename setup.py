from setuptools import setup, find_packages

_version = "0.2"

INSTALL_REQUIRES = []

with open("requirements.txt", "r") as fh:
    for line in fh:
        INSTALL_REQUIRES.append(line.rstrip())

setup(
    name='dna_bayes',
    version=_version,
    description='Bayesian classification of nucleotide sequences',
    author='remita',
    author_email='amine.m.remita@gmail.com',
    packages=find_packages(),
    #include_package_data=True,
    #scripts=['evaluations/eval_complete_seqs.py', 
    #    'evaluations/eval_fragment_seqs.py',
    #    'evaluations/experiment.py'],
    install_requires=INSTALL_REQUIRES
)

