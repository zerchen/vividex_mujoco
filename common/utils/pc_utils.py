import numpy as np

def trans_pcs(pcs, hand_pos, hand_rot, obj_target, cs):
    if cs == "world":
        return pcs
    elif cs == "hand":
        pcs_list = []
        pcs_list.append(pcs)
        for idx in range(len(hand_pos)):
            trans_pcs = pcs - hand_pos[idx]
            trans_pcs = (hand_rot[idx].reshape((3, 3)).T @ trans_pcs.transpose(1, 0)).transpose(1, 0)
            pcs_list.append(trans_pcs)
        return np.concatenate(pcs_list, axis=1)
    elif cs == "target":
        pcs_list = []
        pcs_list.append(pcs)
        pcs_list.append(pcs - obj_target)
        return np.concatenate(pcs_list, axis=1)
    elif cs == "all":
        pcs_list = []
        pcs_list.append(pcs)
        pcs_list.append(pcs - obj_target)
        for idx in range(len(hand_pos)):
            trans_pcs = pcs - hand_pos[idx]
            trans_pcs = (hand_rot[idx].reshape((3, 3)).T @ trans_pcs.transpose(1, 0)).transpose(1, 0)
            pcs_list.append(trans_pcs)
        return np.concatenate(pcs_list, axis=1)
