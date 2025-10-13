from rdkit import Chem
from rdkit.Chem import AllChem

def normalize_protonated_nitrogens(mol):
    """
    For each N with formal charge +1 ([NH+], [NH2+], [NH3+]),
    convert it to [N+] or [n+] (if aromatic) and remove explicit H’s.
    """
    rw_mol = Chem.RWMol(mol)
    for atom in rw_mol.GetAtoms():
        if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1:
            # Keep aromatic status if it is aromatic
            if atom.GetIsAromatic():
                atom.SetIsAromatic(True)  # => [n+]
            else:
                atom.SetIsAromatic(False) # => [N+]
            # Remove explicit hydrogens (the formal charge stays +1)
            atom.SetNumExplicitHs(0)
            atom.UpdatePropertyCache()
    return rw_mol.GetMol()

def find_protonated_n_indices(mol):
    """
    Returns the list of indices of N atoms with formal charge +1.
    """
    indices = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1:
            indices.append(atom.GetIdx())
    return indices

def count_bond_orders(n_atom):
    """
    Standard bond-order sum for a non-aromatic N:
      SINGLE => 1
      DOUBLE => 2
      TRIPLE => 3
      AROMATIC => 1.5 (in standard RDKit)
    We'll only use this if N is NOT aromatic.
    """
    total = 0.0
    for bond in n_atom.GetBonds():
        total += bond.GetBondTypeAsDouble()  # e.g. SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=1.5
    return total

def get_open_sites_on_n_plus(n_atom):
    """
    Custom logic for how many more substituents a [N+] can accept:
      - If the N+ is aromatic, we treat the ring portion as '3 bonds used' 
        (representing 1 single + 1 double in resonance).
        Then we add the bond orders of any substituents that are 
        outside the ring (or if the ring actually has extra neighbors).
      - Else, sum all bond orders in the usual way.
      - open_sites = 4 - total_used
      - if open_sites < 0, set to 0.
    """
    if n_atom.GetIsAromatic():
        # Start with 3 used for the ring (1 single + 1 double).
        # Then add any extra substituent bond(s).
        ring_used = 3.0
        # Let’s see how many neighbors exist in total
        # and subtract the ring-portion from the total if needed.
        # A simple approach: 
        #   total_bond_orders = sum bond orders
        #   total_used = max(ring_used, total_bond_orders)
        # Because if the ring somehow is 2 ring bonds, 
        # that might be 1.5 + 1.5 = 3 anyway in standard RDKit for aromatic.
        total_bond_orders = 0.0
        for bond in n_atom.GetBonds():
            total_bond_orders += bond.GetBondTypeAsDouble()
        total_used = max(ring_used, total_bond_orders)
    else:
        # Non-aromatic N+: sum actual bond orders
        total_used = count_bond_orders(n_atom)

    open_sites = 4 - total_used
    if open_sites < 0:
        open_sites = 0
    return int(open_sites)

def merge_tail_onto_core(core_mol, core_n_idx, tail_mol):
    """
    Merge the tail's [N+] onto the core’s [N+] at index core_n_idx.
    The tail's own [N+] is removed, and its neighbors connect to the core N.
    Returns new RDKit Mol.
    """
    # 1) Identify the tail’s [N+]
    tail_n_idx = None
    for atom in tail_mol.GetAtoms():
        if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1:
            tail_n_idx = atom.GetIdx()
            break
    if tail_n_idx is None:
        raise ValueError("No protonated nitrogen found in tail!")

    rw_core = Chem.RWMol(core_mol)
    rw_tail = Chem.RWMol(tail_mol)

    tail_n_atom = rw_tail.GetAtomWithIdx(tail_n_idx)
    tail_neighbors = tail_n_atom.GetNeighbors()

    # 2) Copy all tail atoms (except for the tail’s N+) into the core
    tail_to_core_map = {}
    for atom in rw_tail.GetAtoms():
        idx = atom.GetIdx()
        if idx == tail_n_idx:
            continue
        new_idx = rw_core.AddAtom(atom)
        tail_to_core_map[idx] = new_idx

    # 3) Copy all bonds in the tail (except those involving the tail’s N+)
    for bond in rw_tail.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if tail_n_idx in (a1, a2):
            continue  # skip the bond to tail’s N
        new_a1 = tail_to_core_map[a1]
        new_a2 = tail_to_core_map[a2]
        rw_core.AddBond(new_a1, new_a2, bond.GetBondType())

    # 4) Now connect the tail N's neighbors to the core’s N
    for nbr in tail_neighbors:
        if nbr.GetIdx() == tail_n_idx:
            continue
        new_nbr_idx = tail_to_core_map[nbr.GetIdx()]
        rw_core.AddBond(core_n_idx, new_nbr_idx, Chem.BondType.SINGLE)

    return rw_core.GetMol()

def attach_tails_to_core(core_smiles, tail_smiles_list):
    """
    Main function:
      1) Parse & normalize core (replace [NH+], [NH2+], [NH3+] with [N+] or [n+]).
      2) Find all N+ in the core.
      3) For each N+, compute open sites based on your custom rule:
         - If aromatic, treat ring usage as 3 => can have 1 open site unless there's another substituent.
         - Else, sum bond orders normally.
      4) If open sites > 0, attach that many tails (or as many tails as we have).
      5) Return the final SMILES.
    """
    core_mol = Chem.MolFromSmiles(core_smiles)
    if core_mol is None:
        raise ValueError(f"Invalid core SMILES: {core_smiles}")
    # Normalize the protonated Ns
    core_mol = normalize_protonated_nitrogens(core_mol)

    rw_mol = Chem.RWMol(core_mol)

    # 2) Identify all N+ in ascending index order
    n_plus_indices = find_protonated_n_indices(rw_mol.GetMol())
    n_plus_indices.sort()

    tail_idx = 0  # pointer in the tail list
    for n_idx in n_plus_indices:
        # Double-check still [N+]
        n_atom = rw_mol.GetAtomWithIdx(n_idx)
        if n_atom.GetAtomicNum() != 7 or n_atom.GetFormalCharge() != 1:
            continue

        # 3) Compute how many open sites remain
        K = get_open_sites_on_n_plus(n_atom)
        if K <= 0:
            continue

        # 4) Attach up to K tails (if available)
        if tail_idx >= len(tail_smiles_list):
            break  # no more tails
        chosen_tails = tail_smiles_list[tail_idx : tail_idx + K]
        tail_idx += len(chosen_tails)  # might be fewer than K if we run out

        for t_sm in chosen_tails:
            tail_mol = Chem.MolFromSmiles(t_sm)
            if tail_mol is None:
                raise ValueError(f"Invalid tail SMILES: {t_sm}")
            tail_mol = normalize_protonated_nitrogens(tail_mol)

            merged = merge_tail_onto_core(rw_mol.GetMol(), n_idx, tail_mol)
            rw_mol = Chem.RWMol(merged)

    final_mol = rw_mol.GetMol()
    Chem.SanitizeMol(final_mol)
    AllChem.Compute2DCoords(final_mol)
    return Chem.MolToSmiles(final_mol)


if __name__ == "__main__":
    # Example:
    A = "c1ccc[nH+]c1"  # a protonated pyridine-like ring
    B = [
        "CCCCCCCCCCCC[NH3+]", 
        "CCCCCCOC(=O)C[NH3+]", 
        "CCCCC[NH3+]", 
        "CCCCC[NH3+]"
    ]
    result_smiles = attach_tails_to_core(A, B)
    print("Final molecule:", result_smiles)

