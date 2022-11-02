from multistrand.objects import *
from multistrand.options import Options
from multistrand.system import *

TEMPERATURE=37
unimolecular_scaling = 5.6e6
bimolecular_scaling = 5.6e6


o = Options(temperature=TEMPERATURE,dangles="Some",substrate_type="RNA",
    bimolecular_scaling=bimolecular_scaling,unimolecular_scaling=unimolecular_scaling)

initialize_energy_model(o)


Loop_Energy = 0    # argument value to energy() function call, below, requesting no dG_assoc or dG_volume terms to be added.  So only loop energies remain.
Volume_Energy = 1  # argument value to energy() function call, below, requesting dG_volume but not dG_assoc terms to be added.  No clear interpretation for this.
Complex_Energy = 2 # argument value to energy() function call, below, requesting dG_assoc but not dG_volume terms to be added.  This is the NUPACK complex microstate energy.
Tube_Energy = 3    # argument value to energy() function call, below, requesting both dG_assoc and dG_volume terms to be added.  Summed over complexes, this is the system state energy.

# Input different sequences and structures
SEQUENCE_TOT = ["ACUGAUCGUAGUCAC","AUUGAGCAUAUUCAC","CGGGCUAUUUAGCUG"] # I1,I2,I3
SEQUENCE = SEQUENCE_TOT[2] # I1
# STRUCTURE = len(SEQUENCE) * '.'
STRUCTURE = "..((((....))))."


c = Complex( strands=[Strand(name="hairpin", sequence=SEQUENCE)], structure= STRUCTURE )
print energy( [c], o, Complex_Energy)  # should be -1.1449...
# Note that energy() takes a *list* of complexes, and returns a tuple of energies.  Don't give it just a complex as input, else all hell may break loose.


import nupack
MATERIAL = 'rna'
dna_seqs = [SEQUENCE]
print nupack.energy(dna_seqs, STRUCTURE, material = MATERIAL, T=TEMPERATURE)
