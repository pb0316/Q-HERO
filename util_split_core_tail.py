from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit

def try_n_plus_or_N_plus(base_mol: Chem.Mol) -> str:
    """
    Given a molecule `base_mol` that presumably has exactly one
    quaternary nitrogen [N+]/n+, produce a base SMILES.
    
    Then create two candidate SMILES:
      (1) Replacing every '[N+]' with 'n+'  (the "aromatic" guess)
      (2) Replacing every 'n+'   with '[N+]' (the "aliphatic" guess)
    
    Attempt to sanitize each. If one passes, return that SMILES.
    If both pass, you can decide which to prefer (here we return the first that passes).
    If both fail, return the original base SMILES.
    """
    base_smi = Chem.MolToSmiles(base_mol, isomericSmiles=True)
    
    # Candidate 1: try "n+" in place of "[N+]"
    candidate1_smi = base_smi.replace("[N+]", "n+")
    candidate1_mol = Chem.MolFromSmiles(candidate1_smi)
    if candidate1_mol is not None:
        try:
            Chem.SanitizeMol(candidate1_mol)
            return Chem.MolToSmiles(candidate1_mol, isomericSmiles=True)
        except Exception:
            pass
    
    # Candidate 2: try "[N+]" in place of "n+"
    candidate2_smi = base_smi.replace("n+", "[N+]")
    candidate2_mol = Chem.MolFromSmiles(candidate2_smi)
    if candidate2_mol is not None:
        try:
            Chem.SanitizeMol(candidate2_mol)
            return Chem.MolToSmiles(candidate2_mol, isomericSmiles=True)
        except Exception:
            pass
    
    # If both fail, return the original base SMILES
    return base_smi


def has_min_carbon_chain(mol: Chem.Mol, n: int) -> bool:
    """
    Returns True if `mol` contains a (linear) chain of at least `n` carbons in a row.
    We construct an RDKit SMARTS that looks like: C-C-C-... (n times).
    """
    if mol is None:
        return False
    # e.g. for n=6: pattern = "C-C-C-C-C-C"
    pattern_str = "-".join(["C"] * n)
    pattern = Chem.MolFromSmarts(pattern_str)
    return mol.HasSubstructMatch(pattern)


def cut_once(
    smi: str, 
    min_carbons: int = 6
):
    """
    Attempt to cut the molecule 'smi' at exactly ONE [N+] substituent that:
      - contains no extra [N+],
      - and after removing that [N+], has at least `min_carbons` in a linear chain.

    Returns (sub_frag_smi, remainder_smi) if success,
            (None, None) if no valid cut is found.
    
    Uses the 'try_n_plus_or_N_plus' function to pick the correct N+ label in each piece.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return (None, None)

    # Find all [N+] atoms
    n_plus_indices = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 7 and a.GetFormalCharge() == 1
    ]
    if not n_plus_indices:
        return (None, None)
    
    for n_idx in n_plus_indices:
        neighbor_indices = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(n_idx).GetNeighbors()]
        
        for nb_idx in neighbor_indices:
            # BFS for that neighbor
            visited = set([nb_idx])
            queue = [nb_idx]
            while queue:
                curr = queue.pop(0)
                curr_atom = mol.GetAtomWithIdx(curr)
                for nbr in curr_atom.GetNeighbors():
                    nbr_idx = nbr.GetIdx()
                    if nbr_idx == n_idx:
                        continue
                    if nbr_idx not in visited:
                        visited.add(nbr_idx)
                        queue.append(nbr_idx)
            
            # Check if BFS substituent has another N+
            has_other_nplus = any(
                (mol.GetAtomWithIdx(a).GetAtomicNum() == 7 and 
                 mol.GetAtomWithIdx(a).GetFormalCharge() == 1)
                for a in visited
            )
            if has_other_nplus:
                continue
            
            # Build sub-mol with BFS + the chosen [N+]
            atoms_to_keep_sub = visited.union({n_idx})
            rwmol_sub = Chem.RWMol(mol)
            to_remove_sub = sorted(
                [a.GetIdx() for a in rwmol_sub.GetAtoms() 
                 if a.GetIdx() not in atoms_to_keep_sub],
                reverse=True
            )
            for x in to_remove_sub:
                rwmol_sub.RemoveAtom(x)
            sub_mol = rwmol_sub.GetMol()
            
            # Temporarily remove [N+] from sub_mol to check if it has an n-carbon chain
            n_plus_in_sub = [
                a.GetIdx() for a in sub_mol.GetAtoms()
                if a.GetAtomicNum() == 7 and a.GetFormalCharge() == 1
            ]
            if len(n_plus_in_sub) != 1:
                continue
            rwmol_for_chain = Chem.RWMol(sub_mol)
            rwmol_for_chain.RemoveAtom(n_plus_in_sub[0])
            sub_for_chain = rwmol_for_chain.GetMol()
            
            # Check if sub_for_chain has at least `min_carbons` in a row
            if not has_min_carbon_chain(sub_for_chain, min_carbons):
                continue
            
            # Build the remainder (everything except this BFS substituent minus the N index)
            atoms_to_keep_rem = set(range(mol.GetNumAtoms()))
            for vid in visited:
                if vid != n_idx:
                    atoms_to_keep_rem.discard(vid)
            
            rwmol_rem = Chem.RWMol(mol)
            to_remove_rem = sorted(
                [ix for ix in range(mol.GetNumAtoms()) 
                 if ix not in atoms_to_keep_rem],
                reverse=True
            )
            for x in to_remove_rem:
                rwmol_rem.RemoveAtom(x)
            remainder_mol = rwmol_rem.GetMol()
            
            # Fix each piece with "try_n_plus_or_N_plus"
            sub_frag_smi = try_n_plus_or_N_plus(sub_mol)
            remainder_smi = try_n_plus_or_N_plus(remainder_mol)
            
            return (sub_frag_smi, remainder_smi)

    return (None, None)


def cut_once_main(smi: str):
    """
    Tries to cut 'smi' using 6+ carbons first.
    If that fails, tries 4+ carbons.
    Returns (sub_frag_smi, remainder_smi) or (None, None).
    """
    subfrag_6, remainder_6 = cut_once(smi, min_carbons=6)
    if subfrag_6 is not None:
        return (subfrag_6, remainder_6)
    
    # Fall back to 4+ carbons
    return cut_once(smi, min_carbons=4)


def iterative_cut(smi: str, results=None):
    """
    Repeatedly apply `cut_once_main` until no more valid cuts remain.
    Each cut yields:
      - A sub-fragment (the BFS substituent + N+),
      - The remainder, which we keep cutting.
    Once we cannot cut further, we add that final 'smi' as a fragment.
    """
    if results is None:
        results = []
    
    subfrag, remainder = cut_once_main(smi)
    if subfrag is None:
        # No more cuts => add 'smi' as final (with correct labeling)
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            final_smi = try_n_plus_or_N_plus(mol)
            results.append(final_smi)
        else:
            results.append(smi)
    else:
        # We made a cut => store the BFS subfragment
        results.append(subfrag)
        # Recursively cut the remainder
        iterative_cut(remainder, results)

    return results


# ------------------- EXAMPLE USAGE ------------------- #
if __name__ == "__main__":
    test_smi = "C[N+](C=C1)=CC=C1C2=CC=[N+](CCCCCCCCCCCCCCCC)C=C2.[Br-].[I-]"
    
    final_fragments = iterative_cut(test_smi)
    
    print("Fragments from iterative one-cut-at-a-time decomposition:")
    for i, frag in enumerate(final_fragments, 1):
        print(f"{i}. {frag}")
