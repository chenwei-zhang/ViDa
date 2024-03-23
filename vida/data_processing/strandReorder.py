import re
from itertools import permutations
import numpy as np


class ThreeStrandReorder:
    
    @staticmethod
    def get_strand_pos(bpair):
        pattern = r'([a-z])(\d+)([a-z])(\d+)'
        match = re.match(pattern, bpair)

        if match:
            strand1, pos1, strand2, pos2 = match.groups()
            pos1 = int(pos1) - 1  # Convert to 0-based index
            pos2 = int(pos2) - 1  # Convert to 0-based index
            return strand1, pos1, strand2, pos2
        else:
            raise ValueError("Invalid base pair format")


    @staticmethod
    def hasIncbInvPair(base_pairs):
        has_bc = any('b' in pair and 'c' in pair for pair in base_pairs)

        return 1 if has_bc else 0
    
    
    def get_basepairs(self, seq, dp, ref_name_list, strand_list):
        def concat_disorder(seq, ref_name_list, strand_list):
            sequence_list = re.split(r'\s|\+', seq) 
            sequence = ''.join(sequence_list)

            for permuted_strand in permutations(strand_list):
                combined_sequence = ''.join(permuted_strand)
                if combined_sequence == sequence:
                    alter_name = [ref_name_list[strand_list.index(strand)] for strand in permuted_strand]
                    break

            return np.concatenate(alter_name)

        dp_structure = dp.replace(' ', '').replace('+', '')
        alter_name = concat_disorder(seq, ref_name_list, strand_list)

        stack = []
        base_pairs = []

        for name, char in zip(alter_name, dp_structure):
            if char == '(':
                stack.append(name)
            elif char == ')':
                if stack:
                    opening_index = stack.pop()
                    base_pairs.append(opening_index + name)
                else:
                    raise ValueError("Mismatched brackets")

        if stack:
            raise ValueError("Mismatched brackets")

        return base_pairs


    def reorderCase1(self, case1, dp, seq, short_seqname, ref_name_list, strand_list):
        """
        Reorder:
        'incb+inv+sub' -> 'inv+sub+incb' 
        'sub+incb+inv' -> 'inv+sub+incb'
        """
        
        position = case1.index(short_seqname)
        base_pairs = self.get_basepairs(seq, dp, ref_name_list, strand_list)
        
        incb_inv_pair = ThreeStrandReorder.hasIncbInvPair(base_pairs)
                
        if position == 0:   # 'inv+sub+incb': c+a+b ===> do nothing
            return dp, short_seqname, incb_inv_pair
        
        
        if position == 1:  # 'incb+inv+sub': b+c+a ===> c+a+b
            incb_list, inv_list, sub_list = [list(part) for part in re.split(r'\s|\+', dp)]
            
            for bpair in base_pairs:
                strand1, pos1, strand2, pos2 = ThreeStrandReorder.get_strand_pos(bpair)
                
                # only need to change incb (b) with its connected strands
                if strand1 == 'b' and strand2 == 'c':
                    incb_list[pos1] = ')'
                    inv_list[pos2] = '('

                if strand1 == 'b' and strand2 == 'a':
                    incb_list[pos1] = ')'
                    sub_list[pos2] = '('


        if position == 2: # 'sub+incb+inv': a+b+c ===> c+a+b
            sub_list, incb_list, inv_list = [list(part) for part in re.split(r'\s|\+', dp)]
            
            for bpair in base_pairs:
                strand1, pos1, strand2, pos2 = ThreeStrandReorder.get_strand_pos(bpair)
            
                # only need to change inv (c) with its connected strands
                if strand1 == 'a' and strand2 == 'c':
                    inv_list[pos2] = '('
                    sub_list[pos1] = ')'
                                    
                if strand1 == 'b' and strand2 == 'c':
                    inv_list[pos2] = '('
                    incb_list[pos1] = ')'
        
        inv = ''.join(inv_list)
        sub = ''.join(sub_list)
        incb = ''.join(incb_list)
        # inv+sub+incb
        dp_new = inv + "+" + sub + "+" + incb
        
        return dp_new, case1[0], incb_inv_pair


    def reorderCase2(self, case2, dp, seq, short_seqname, ref_name_list, strand_list):
        """
        Reorder:
        'inv+incb+sub' -> 'incb+sub+inv' 
        'sub+inv+incb' -> 'incb+sub+inv'
        """
        
        position = case2.index(short_seqname)
        base_pairs = self.get_basepairs(seq, dp, ref_name_list, strand_list)
        
        incb_inv_pair = ThreeStrandReorder.hasIncbInvPair(base_pairs)
                
        if position == 0:   # 'incb+sub+inv': b+a+c ===> do nothing
            return dp, short_seqname, incb_inv_pair
        
        if position == 1:  # 'inv+incb+sub': c+b+a ===> b+a+c
            inv_list, incb_list, sub_list = [list(part) for part in re.split(r'\s|\+', dp)]
            
            for bpair in base_pairs:
                strand1, pos1, strand2, pos2 = ThreeStrandReorder.get_strand_pos(bpair)
                
                # only need to change inv (c) with its connected strands
                if strand1 == 'c' and strand2 == 'b':
                    inv_list[pos1] = ')'
                    incb_list[pos2] = '('
                    incb_inv_pair = 1
                    
                if strand1 == 'c' and strand2 == 'a':
                    inv_list[pos1] = ')'
                    sub_list[pos2] = '('
        
        
        if position == 2: # 'sub+inv+incb': a+c+b ===> b+a+c
            sub_list, incb_list, inv_list = [list(part) for part in re.split(r'\s|\+', dp)]
            
            for bpair in base_pairs:
                strand1, pos1, strand2, pos2 = ThreeStrandReorder.get_strand_pos(bpair)
            
                # only need to change incb (b) with its connected strands
                if strand1 == 'a' and strand2 == 'b':
                    incb_list[pos2] = '('
                    sub_list[pos1] = ')'
                    
                if strand1 == 'c' and strand2 == 'b':
                    incb_list[pos2] = '('
                    inv_list[pos1] = ')'
        
        inv = ''.join(inv_list)
        sub = ''.join(sub_list)
        incb = ''.join(incb_list)
        # incb+sub+inv
        dp_new = incb + "+" + sub + "+" + inv
        
        return dp_new, case2[0], incb_inv_pair


    def reorderCase3(self, case3, dp, seq, short_seqname, ref_name_list, strand_list):
        """
        Reorder:
        'sub+incb inv' -> 'incb+sub inv' 
        'inv sub+incb' -> 'incb+sub inv'
        'sub+inv incb' -> 'incb+sub inv' 
        """
        
        position = case3.index(short_seqname)
        
        incb_inv_pair = 0 # certainly no incumbent-invader pair
                
        if position == 0:   # 'incb+sub inv': b+a c ===> do nothing
            return dp, short_seqname, incb_inv_pair
        
        
        base_pairs = self.get_basepairs(seq, dp, ref_name_list, strand_list)
        
        if position == 1:  # 'sub+incb inv': a+b c ===> b+a c
            sub_list, incb_list, inv_list  = [list(part) for part in re.split(r'\s|\+', dp)]
            
            for bpair in base_pairs:
                strand1, pos1, strand2, pos2 = ThreeStrandReorder.get_strand_pos(bpair)
                
                # only need to change incb (b) with its connected strand sub (a)
                if strand1 == 'a' and strand2 == 'b':
                    incb_list[pos2] = '('
                    sub_list[pos1] = ')'
        
        
        if position == 2: # 'inv sub+incb': c a+b ===> b+a c
            inv_list, sub_list, incb_list = [list(part) for part in re.split(r'\s|\+', dp)]
            
            for bpair in base_pairs:
                strand1, pos1, strand2, pos2 = ThreeStrandReorder.get_strand_pos(bpair)
                
                # only need to change incb (b) with its connected strand sub (a)
                if strand1 == 'a' and strand2 == 'b':
                    incb_list[pos2] = '('
                    sub_list[pos1] = ')'
                    
        
        if position == 3: # 'inv incb+sub': c b+a ===> b+a c
            inv_list, incb_list, sub_list = [list(part) for part in re.split(r'\s|\+', dp)]
            # just need to exchange the position of inv and incb+sub
            
        inv = ''.join(inv_list)
        sub = ''.join(sub_list)
        incb = ''.join(incb_list)
        # incb+sub inv
        dp_new = incb + "+" + sub + " " + inv
        
        return dp_new, case3[0], incb_inv_pair


    def reorderCase4(self, case4, dp, seq, short_seqname, ref_name_list, strand_list):
        """
        Reorder:
        'incb inv+sub' -> 'incb sub+inv' 
        'inv+sub incb' -> 'incb sub+inv' 
        'sub+inv incb' -> 'incb sub+inv' 
        """
        
        position = case4.index(short_seqname)
        
        incb_inv_pair = 0 # certainly no incumbent-invader pair
                
        if position == 0:   # 'incb sub+inv': b a+c ===> do nothing
            return dp, short_seqname, incb_inv_pair
        
        
        base_pairs = self.get_basepairs(seq, dp, ref_name_list, strand_list)
        
        if position == 1:  # 'incb inv+sub': b c+a ===> b a+c
            incb_list, inv_list, sub_list = [list(part) for part in re.split(r'\s|\+', dp)]
            
            for bpair in base_pairs:
                strand1, pos1, strand2, pos2 = ThreeStrandReorder.get_strand_pos(bpair)
                
                # only need to change inv (c) with its connected strand sub (a)
                if strand1 == 'c' and strand2 == 'a':
                    inv_list[pos1] = ')'
                    sub_list[pos2] = '('
                    
        
        if position == 2: # 'inv+sub incb': c+a b ===> b a+c
            inv_list, sub_list, incb_list = [list(part) for part in re.split(r'\s|\+', dp)]
            
            for bpair in base_pairs:
                strand1, pos1, strand2, pos2 = ThreeStrandReorder.get_strand_pos(bpair)
                
                # only need to change inv (c) with its connected strand sub (a)
                if strand1 == 'c' and strand2 == 'a':
                    inv_list[pos1] = ')'
                    sub_list[pos2] = '('
        
        
        if position == 3: # 'sub+inv incb': a+c b ===> b a+c
            sub_list, inv_list, incb_list = [list(part) for part in re.split(r'\s|\+', dp)]
            # just need to exchange the position of sub+inv and incb
            
        inv = ''.join(inv_list)
        sub = ''.join(sub_list)
        incb = ''.join(incb_list)
        # incb sub+inv
        dp_new = incb + " " + sub + "+" + inv
        
        return dp_new, case4[0], incb_inv_pair


    def reorderCase5(self, case5, dp, seq, short_seqname, ref_name_list, strand_list):
        """
        Reorder:
        'inv+incb sub' -> 'incb+inv sub' 
        'sub inv+incb' -> 'incb+inv sub' 
        'sub incb+inv' -> 'incb+inv sub' 
        """
        
        position = case5.index(short_seqname)
        
        incb_inv_pair = 1 # certainly having incumbent-invader pair
                
        if position == 0:   # 'incb+inv sub': b+c a ===> do nothing
            return dp, short_seqname, incb_inv_pair
        
        
        base_pairs = self.get_basepairs(seq, dp, ref_name_list, strand_list)
        
        if position == 1:  # 'inv+incb sub': c+b a ===> b+c a
            inv_list, incb_list, ub_list = [list(part) for part in re.split(r'\s|\+', dp)]
            
            for bpair in base_pairs:
                strand1, pos1, strand2, pos2 = ThreeStrandReorder.get_strand_pos(bpair)
                
                # only need to change inv (c) with its connected strand incb (b)
                if strand1 == 'c' and strand2 == 'b':
                    inv_list[pos1] = ')'
                    incb_list[pos2] = '('
                    
        
        if position == 2: # 'sub inv+incb': a c+b ===> b+c a
            sub_list, inv_list, incb_list = [list(part) for part in re.split(r'\s|\+', dp)]
            
            for bpair in base_pairs:
                strand1, pos1, strand2, pos2 = ThreeStrandReorder.get_strand_pos(bpair)
                
                # only need to change inv (c) with its connected strand incb (b)
                if strand1 == 'c' and strand2 == 'b':
                    inv_list[pos1] = ')'
                    incb_list[pos2] = '('
        
        
        if position == 3: # 'sub incb+inv': a b+c ===> b+c a
            sub_list, incb_list, inv_list = [list(part) for part in re.split(r'\s|\+', dp)]
            # just need to exchange the position of sub and incb+inv
            
        inv = ''.join(inv_list)
        sub = ''.join(sub_list)
        incb = ''.join(incb_list)
        # incb+inv sub
        dp_new = incb + "+" + inv + " " + sub
        
        return dp_new, case5[0], incb_inv_pair


    def reorderCase6(self, case6, dp, short_seqname):
        """
        Reorder:
        'incb inv sub' -> 'incb sub inv' 
        'sub inv incb' -> 'incb sub inv' 
        'sub incb inv' -> 'incb sub inv' 
        'inv incb sub' -> 'incb sub inv' 
        'inv sub incb' -> 'incb sub inv' 
        """
        
        position = case6.index(short_seqname)
        
        incb_inv_pair = 0 # certainly not incumbent-invader pair
                
        if position == 0:   # 'incb sub inv': b a c ===> do nothing
            return dp, short_seqname, incb_inv_pair
            
        if position == 1:  # 'incb inv sub': b c a ===> b a c
            incb_list, inv_list, sub_list = [list(part) for part in re.split(r'\s|\+', dp)]
        
        if position == 2: # 'sub inv incb': a c b ===> b a c
            sub_list, inv_list, incb_list = [list(part) for part in re.split(r'\s|\+', dp)]
        
        if position == 3: # 'sub incb inv': a b c ===> b a c
            sub_list, incb_list, inv_list = [list(part) for part in re.split(r'\s|\+', dp)]

        if position == 4: # 'inv incb sub': c b a ===> b a c
            inv_list, incb_list, sub_list = [list(part) for part in re.split(r'\s|\+', dp)]
        
        if position == 5: # 'inv sub incb': c a b ===> b a c
            inv_list, sub_list, incb_list = [list(part) for part in re.split(r'\s|\+', dp)]
                
        inv = ''.join(inv_list)
        sub = ''.join(sub_list)
        incb = ''.join(incb_list)
        # incb sub inv
        dp_new = incb + " " + sub + " " + inv
        
        return dp_new, case6[0], incb_inv_pair
    
    
    def nameMap(self, sequence, strand_sub, strand_incb, strand_inv):
        def replace_strings(s):
            s = s.replace(strand_inv, "inv")
            s = s.replace(strand_incb, "incb")
            s = s.replace(strand_sub, "sub")
            return s

        vectorized_replace = np.vectorize(replace_strings)

        return vectorized_replace(np.array([sequence])).tolist()[0]
    
    
    def dp_reorder(self, dp, seq, ref_name_list, strand_list, strand_sub, strand_incb, strand_inv):
    
        short_seqname = self.nameMap(seq, strand_sub, strand_incb, strand_inv)
        
        case1 = ['inv+sub+incb', 'incb+inv+sub', 'sub+incb+inv']
        case2 = ['incb+sub+inv', 'inv+incb+sub', 'sub+inv+incb']
        case3 = ['incb+sub inv', 'sub+incb inv', 'inv sub+incb', 'inv incb+sub'] 
        case4 = ['incb sub+inv', 'incb inv+sub', 'inv+sub incb', 'sub+inv incb'] 
        case5 = ['incb+inv sub', 'inv+incb sub', 'sub inv+incb', 'sub incb+inv'] ##
        case6 = ['incb sub inv', 'incb inv sub', 'sub inv incb' 'sub incb inv' 'inv incb sub' 'inv sub incb'] ##
        
        if short_seqname in case1:
            dp_new, short_seqname, incb_inv_pair = self.reorderCase1(case1, dp, seq, short_seqname, ref_name_list, strand_list)
            
        if short_seqname in case2:
            dp_new, short_seqname, incb_inv_pair = self.reorderCase2(case2, dp, seq, short_seqname, ref_name_list, strand_list)
        
        if short_seqname in case3:
            dp_new, short_seqname, incb_inv_pair = self.reorderCase3(case3, dp, seq, short_seqname, ref_name_list, strand_list)
            
        if short_seqname in case4:
            dp_new, short_seqname, incb_inv_pair = self.reorderCase4(case4, dp, seq, short_seqname, ref_name_list, strand_list)
        
        if short_seqname in case5:
            dp_new, short_seqname, incb_inv_pair = self.reorderCase5(case5, dp, seq, short_seqname, ref_name_list, strand_list)
        
        if short_seqname in case6:
            dp_new, short_seqname, incb_inv_pair = self.reorderCase6(case6, dp, seq, short_seqname, ref_name_list, strand_list)
        
        return dp_new, short_seqname, incb_inv_pair
