from functools import partial
import src.finetune.utils.inconsistencies as inc
import numpy as np
import torch

def resolve_consistency1(solve, voxel_ids, object_ids, object_id_to_logits):
        
    consistency_solution = partial(
        inc.solve_inconsistency3,  # main inconsistency solution function
        solve_function=solve,  # `solve_function` is used in solve_inconsistency
        voxel_ids=voxel_ids,
        object_ids=object_ids,
        object_id_to_logits=object_id_to_logits,
    )
    return consistency_solution
        
        
def resolve_consistency2(solve, voxel_ids, update_voxels, object_ids,  object_id_to_logits):
    '''
    time improved func. First compute all composition of object id. Record and Search
    Reduce the repeated computation of consistency of voxel
    '''
    def get_ids(vox_ids, object_ids, voxel_ids):
    
        part_len = len(vox_ids)
        all_len = len(voxel_ids)
        vox_ids = vox_ids[np.newaxis,...].T.repeat(all_len, 1)  # len(vox_ids), len(voxel_ids)
        voxel_ids = voxel_ids[np.newaxis,...].repeat(part_len, 0)
        object_ids = object_ids[np.newaxis, ...].repeat(part_len, 0)
        mask = voxel_ids == vox_ids     # lid_cpuen(vox_ids), len(voxel_ids)


        id_couples = []
        for i in range(part_len):
            objects = object_ids[i, :][mask[i, :]]
            ids = tuple(np.unique(objects, return_counts=True)[0])
            key = tuple(sorted(ids)) 
            id_couples.append(key)  
        return id_couples

    id_couples = get_ids(update_voxels, object_ids, voxel_ids)
    unique_id_cps = set(id_couples)

    solve_function=solve   #_ours_impl
    
    computed_dict = {}
    for uic in unique_id_cps:
        logits = []
        for i in uic:
            obj_logits = object_id_to_logits[i].cpu()
            if len(obj_logits.shape) == 1:
                obj_logits = obj_logits.unsqueeze(0)
            logits.append(obj_logits)
        logits = torch.cat(logits)
        resolved_class, _ = solve_function(logits)
        computed_dict[uic]=(resolved_class, logits)

    def get_inconsistency(
        vox_ids, object_ids, voxel_ids, inconsis_dict
    ):
        results = []
        for vox_id in vox_ids:
            objects = object_ids[voxel_ids == vox_id]
            if len(objects) == 0:
                return None, None
            ids = tuple(np.unique(objects, return_counts=True)[0])

            key = tuple(sorted(ids))
            resolved_class, logits = inconsis_dict[key]
            results.append((vox_id, (resolved_class, logits)))

        return results
    
    consistency_solution = partial(
            get_inconsistency,  # main inconsistency solution function
            # solve_function=_ours_impl,  # `solve_function` is used in solve_inconsistency
            voxel_ids=voxel_ids,
            object_ids=object_ids,
            # object_id_to_logits=object_id_to_logits,
            inconsis_dict=computed_dict
        )
    return consistency_solution


def resolve_consistency3(update_voxels, 
                         solve, voxel_ids, 
                         object_ids, object_id_to_logits, 
                         inputs):
    '''
       solve: logits solve function
       voxel_ids: all voxels
       object_ids: for all points
       object_id_to_logits: logits for every object
       inputs: update voxels
    '''
    def get_object_ids(
        vox_ids, object_ids, voxel_ids
    ):
        '''
        find interset object_ids in all voxels for update voxels
        
        vox_ids: voxels to be update
        return:
            List[ tuple(object ids), ...] object ids for updated voxels
        '''
        # computed_dict = {}  # 记录voxel里点云的object id一样的计算结果，减少计算时间
        results = []
        for vox_id in vox_ids:
            objects = object_ids[voxel_ids == vox_id]
            if len(objects) == 0:
                continue
            ids = tuple(np.unique(objects, return_counts=True)[0])
            results.append(tuple(sorted(ids)))

        return results
    
    get_objid_func = partial(
        get_object_ids,  # main inconsistency solution function
        # solve_function=_ours_impl,  # `solve_function` is used in solve_inconsistency
        voxel_ids=voxel_ids,
        object_ids=object_ids,
        # object_id_to_logits=object_id_to_logits,
    )

    # updated voxel中，和all voxel交叉的object id
    id_couples = []
    for i in inputs:
        res = get_objid_func(i)
        for r in res:
            id_couples.append(r)
    # print(id_couples)
    unique_id_cps = set(id_couples)     # unique interset object ids with all voxels of update voxels, ((1,3,4), (0,1,2), ...)

    solve_function=solve   #_ours_impl
    
    computed_dict = {}
    for uic in unique_id_cps:
        logits = []
        for i in uic:
            obj_logits = object_id_to_logits[i].cpu()
            if len(obj_logits.shape) == 1:
                obj_logits = obj_logits.unsqueeze(0)
            logits.append(obj_logits)
        logits = torch.cat(logits)
        resolved_class, _ = solve_function(logits)
        computed_dict[uic]=(resolved_class, logits)

    def get_inconsistency(
        vox_ids, object_ids, voxel_ids, inconsis_dict
    ):
        results = []
        for vox_id in vox_ids:
            objects = object_ids[voxel_ids == vox_id]
            if len(objects) == 0:
                return None, None
            ids = tuple(np.unique(objects, return_counts=True)[0])

            key = tuple(sorted(ids))
            resolved_class, logits = inconsis_dict[key]
            results.append((vox_id, (resolved_class, logits)))

        return results
    
    # 所有 交叉的object id，统一class、logits，重新赋值给每个voxel
    consistency_solution = partial(
            get_inconsistency,  # main inconsistency solution function
            # solve_function=_ours_impl,  # `solve_function` is used in solve_inconsistency
            voxel_ids=voxel_ids,
            object_ids=object_ids,
            # object_id_to_logits=object_id_to_logits,
            inconsis_dict=computed_dict
        )
    return consistency_solution