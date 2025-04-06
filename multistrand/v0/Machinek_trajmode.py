import sys
from multistrand.objects import StopCondition, Complex, Strand
from multistrand.options import Options, Literals
from multistrand.system import SimSystem
from multistrand.concurrent import MergeSim


def print_trajectory(o):
    seqstring=''
    for i in range(len(o.full_trajectory)): # go through each output microstate of the trajectory
        time = o.full_trajectory_times[i]   # time at which this microstate is entered
        states = o.full_trajectory[i]       # this is a list of the complexes present in this tube microstate
        newseqs = []
        for state in states: newseqs += [ state[3] ]   # extract the strand sequences in each complex (joined by "+" for multistranded complexes)
        newseqstring = ' '.join(newseqs)    # make a space-separated string of complexes, to represent the whole tube system sequence
        if not newseqstring == seqstring :
            print(newseqstring)
            seqstring=newseqstring          # because strand order can change upon association of dissociation, print it when it changes
        structs = []
        for state in states: structs += [ state[4] ]   # similarly extract the secondary structures for each complex
        tubestruct = ' '.join(structs)      # give the dot-paren secondary structure for the whole test tube
        dG=0
        for state in states: dG += state[5]
        print(f'{tubestruct} t={time} seconds, dG={dG} kcal/mol')
        sys.stdout.flush()  # Flush the output here


def machinek2014_trajmode(mismatchSelect,toeholdSelect='7nt'):
    # we only allow first step mode at this point.
    
    # these are the sequences we need to build the dot-parens
    incumbent = ""
    target = ""
    invader = ""

    # decide on toehold sequence    
    if toeholdSelect == "7nt" :
        toeholdSeq = "ATGTGGA"  # 7 nt toehold option
    if toeholdSelect == "6nt" :
        toeholdSeq = "ATGTGG" # 6 nt toehold option
    if toeholdSelect == "10nt" :
        toeholdSeq = "ATGTGGAGGG"  # 10 nt toehold option
    
    # determine the incumbent, target and invader sequences
    # FD: copy-pasting supplementary Table 6 directly
    if mismatchSelect == 0 or mismatchSelect == 2 or mismatchSelect == 12 or mismatchSelect == 14  or mismatchSelect == '14C2T':
        incumbent = "TGGTGTTTGTGGGTGTGGTGAGTTTGAGGTTGA"
        target = "CCCTCCACATTCAACCTCAAACTCACC"
        
        if mismatchSelect == 0:  # perfect
            invader = "GGTGAGTTTGAGGTTGA"

        if mismatchSelect == 2:
            invader = "GGTGAGTTTGAGGTTCA"
        
        if mismatchSelect == 12:
            invader = "GGTGACTTTGAGGTTGA"
            
        if mismatchSelect == 14:
            invader = "GGTCAGTTTGAGGTTGA"

        if mismatchSelect == '14C2T':
            invader = "GGTTAGTTTGAGGTTGA"
    
    if mismatchSelect == 3:
        incumbent = "TGGTGTTTGTGGGTGTGGTGAGTTTGAGGTGAT"
        target = "CCCTCCACATATCACCTCAAACTCACC"
        invader = "GGTGAGTTTGAGGTCAT"
        
    if mismatchSelect == 4:
        incumbent = "TGGTGTTTGTGGGTGTGGTGAGTTTGAGTGAGT"
        target = "CCCTCCACATACTCACTCAAACTCACC"
        invader = "GGTGAGTTTGAGTCAGT"

    if mismatchSelect == 5:
        incumbent = "TGGTGTTTGTGGGTGTGGTGAGTTTGATGAGGT"
        target = "CCCTCCACATACCTCATCAAACTCACC"
        invader = "GGTGAGTTTGATCAGGT"

    if mismatchSelect == 6:
        incumbent = "TGGTGTTTGTGGGTGTGGTGAGTTTGTGAAGGT"
        target = "CCCTCCACATACCTTCACAAACTCACC"
        invader = "GGTGAGTTTGTCAAGGT"
        
    if mismatchSelect == 7:
        incumbent = "TGGTGTTTGTGGGTGTGGTGAGTTTTGAGAGGT"
        target = "CCCTCCACATACCTCTCAAAACTCACC"
        invader = "GGTGAGTTTTCAGAGGT"

    if mismatchSelect == 8:
        incumbent = "TGGTGTTTGTGGGTGTGGTGAGTTTGATGAGGT"
        target = "CCCTCCACATACCTCATCAAACTCACC"
        invader = "GGTGAGTTTCATGAGGT"
        
    if mismatchSelect == 9:
        incumbent = "TGGTGTTTGTGGGTGTGGTGAGTTGATTGAGGT"
        target = "CCCTCCACATACCTCAATCAACTCACC"
        invader = "GGTGAGTTCATTGAGGT"
        
    if mismatchSelect == 10:
        incumbent = "TGGTGTTTGTGGGTGTGGTGAGTGATTTGAGGT"
        target = "CCCTCCACATACCTCAAATCACTCACC"
        invader = "GGTGAGTCATTTGAGGT"

    invader = invader + toeholdSeq
    
    # set up the actual complexes
    strandIncumbent = Strand(name="incumbent", sequence=incumbent)
    strandTarget = Strand(name="target", sequence=target)
    strandInvader = Strand(name="invader", sequence=invader)
    
    intialDotParen = '.' * 16 + '(' * 17 + "+" + '.' * 10 + ')' * 17  
    intialInvaderDotParen = '.' * len(invader)
    successDotParen = '.' * 33
    
    initialComplex = Complex(strands=[strandIncumbent, strandTarget], structure=intialDotParen)
    initialInvader = Complex(strands=[strandInvader], structure=intialInvaderDotParen)
    successComplex = Complex(strands=[strandIncumbent], structure=successDotParen)
    
    stopSuccess = StopCondition(Literals.success, [(successComplex, Literals.dissoc_macrostate, 0)])
    
    # actually set the intial and stopping states    
    start_state = [initialComplex, initialInvader]
    stop_conditions = [stopSuccess]         
    
    return start_state, stop_conditions



# set up config
def create_setup(start_state, stop_conditions):
    o = Options(
        simulation_mode="Trajectory",
        substrate_type="DNA",
        num_simulations=1, 
        simulation_time=float('inf'),
        # simulation_time=1e-2,
        dangles="Some", 
        temperature=4, 
        join_concentration = 100e-6, # 100 uM
        gt_enable = False,
        output_interval = 1, # record every # steps
        verbosity=0,
        start_state = start_state,
        stop_conditions = stop_conditions
        )
    o.DNA23Metropolis()
    
    return o
    
 
    
if __name__ == '__main__':
    
    for i in range(50):
        
        mismatchSelect = 12
        
        start_state, stop_conditions = machinek2014_trajmode(mismatchSelect=mismatchSelect)
            
        o = create_setup(start_state, stop_conditions)
        sim = SimSystem(o)
        sim.start()

        # write trajs to file
        stdoutOrigin=sys.stdout         
        sys.stdout = open(f"./raw_data/Machinek-Mismatch{mismatchSelect}-og/Machinek-Mismatch{mismatchSelect}-{i}.txt", "w")
        # sys.stdout = open(f"./temp-{i}.txt", "w")
    
        print_trajectory(o)
        
        sys.stdout.flush()  # Flush the output here
        sys.stdout.close()
        sys.stdout=stdoutOrigin
        
        print(f"Simulation {i} complete; Trajectory length: {len(o.full_trajectory)}")
        sys.stdout.flush()  # Flush the output here

