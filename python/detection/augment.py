import numpy as np

def augment_patch(patch):
    alist = []
    alist.append(patch)
    for i in range(1,4):
        alist.append(np.rot90(patch,i))
        
    pfl = patch[:,::-1]
    alist.append(pfl)
    for i in range(1,4):
        alist.append(np.rot90(pfl,i))
    return alist

def generate_augmented_set(init_set):
	print("Augmented set generation...")
	print("Initial set size: %dx%dx%d" % init_set.shape)
	augmented_set = []
	for p in init_set:
	    augmented_set.append(augment_patch(p))

	augmented_set = np.concatenate(augmented_set)
	print("Final set size: %dx%dx%d" % augmented_set.shape)
	return augmented_set