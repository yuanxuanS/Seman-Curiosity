from itertools import chain

def list_helper_collate(batch):
    return list(chain(*[[elem for elem in elems_list] for elems_list in batch]))

def dict_helper_collate(batch):
    elem = batch[0]
    return [{key: d[key] for key in elem} for d in batch]

