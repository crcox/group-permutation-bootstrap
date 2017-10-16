#!/usr/bin/env python
from __future__ import print_function

import argparse
import progressbar
import nibabel as nib
import nibabel.processing
import numpy as np
import scipy.ndimage
from timeit import default_timer as timer
import os

p = argparse.ArgumentParser()
p.add_argument('reference_img', help='Path to the image to be thresholded.')
p.add_argument('permutations_dir', help='Path to the directory containing the permutations.')
p.add_argument('-m','--mask',type=str,default='',help='.')
p.add_argument('-s','--number-of-subjects',type=int,default=23,help='Number of subjects.')
p.add_argument('-b','--blur',type=int,default=0,help='Blur amount. Doesn''t actually apply the blur, just looks up the right files.')
p.add_argument('-B','--do-blur',type=int,default=0,help='Blur amount. Actually applies the blur.')
p.add_argument('-r','--number-of-random-seeds',type=int,default=100,help='The number of permutations run for each subject.')
p.add_argument('-z','--base-zero-indexes',action='store_true',help='Use this flag if random seeds and subjects are identified with base-zero indexes.')
p.add_argument('-n','--number-of-permutations',type=int,default=100,help='The number of group level maps to generate to threshold the group reference image.')
p.add_argument('-o','--output-filename',type=str,default='perm_map.nii',help='Name of the output file (will be nifti-1 formatted).')
p.add_argument('-P','--preload-all-permutations',action='store_true',help='Load all permutation data into memory at the outset, to cut down on disk I/O overhead. [WARNING: This may use lots of memory!]')
p.add_argument('-W','--log-and-write-permutations',action='store_true',help='Maintain all n computed group permutation maps in memory and write them to disk at then end.')
p.add_argument('-M','--log-permutations',action='store_true',help='Maintain all n computed group permutation maps in memory, which can speed up cluster size search.')
p.add_argument('-p','--pthreshold',type=float,nargs='+',default=[0.05,0.01,0.001],help='The threshold to apply before identifying clusters.')
args = p.parse_args()

start = timer()
# Load the reference image
reference_img = nib.load(args.reference_img)
reference_img_data = reference_img.get_data()

# Loop and compose maps from the permutations
nperm = args.number_of_permutations
max_subject_perm = args.number_of_random_seeds
nsubj = args.number_of_subjects
perm_dir = args.permutations_dir
if args.base_zero_indexes:
    adj = 0
else:
    adj = 1

hdr = []
dat = []
print("Begin loading data:")
bar = progressbar.ProgressBar()
for s in bar(range(nsubj)):
    try:
        fname = "{subject:02d}_dartel_funcres.nii".format(subject = s+adj, blur = args.blur)
        # fname = "{subject:02d}_C.b{blur:d}.nii".format(subject = s+adj, blur = args.blur)
        hdr.append(nib.load(os.path.join(perm_dir,fname)))
    except:
        fname = "r{subject:02d}_dartel.nii.gz".format(subject = s+adj, blur = args.blur)
        hdr.append(nib.load(os.path.join(perm_dir,fname)))

    # tmp = hdr[s].get_data()
    if args.preload_all_permutations:
        dat.append(np.array(hdr[s].get_data()))
    else:
        dat.append(hdr[s].get_data())

end = timer()
print("Data loaded: {seconds:.4f} seconds".format(seconds=end-start))
middle = timer()

if args.mask:
    mask_img = nib.four_to_three(nib.load(args.mask))[0]
    mask_data = mask_img.get_data()
    # If it is a probability map of tissue types, this says treat all voxels
    # that have a probability 0.2 or higher of being grey matter.
    # 0.2 is based on John Ashburner's VBM tutorial (VBMclass15.pdf) and Kurth,
    # Gaser, and Luders (2015, Nature Protocols)
    mask = mask_data > 0.2
    try:
        reference_img_vec = reference_img_data[mask]
    except:
        rmask = nib.processing.resample_from_to(mask_img, reference_img)
        mask_data = rmask.get_data()
        mask = mask_data > 0.2
        reference_img_data = reference_img_data[mask]

    for s in range(nsubj):
        dat[s] = dat[s][mask]

B = np.zeros(reference_img_data.shape, dtype='int16')

if args.log_and_write_permutations or args.log_permutations:
    if args.mask:
        P = np.zeros((len(reference_img_data),nperm), dtype='float')
    else:
        P = np.zeros(reference_img_data.shape+(nperm,), dtype='float')

bar = progressbar.ProgressBar()
for p in bar(range(nperm)):
    sample = np.random.choice(max_subject_perm, nsubj, replace=True)
    for s,r in enumerate(sample):
        if args.mask:
            img_data = dat[s][:,r]
        else:
            try:
                img_data = dat[s][:,:,:,0,r]
            except IndexError:
                img_data = dat[s][:,:,:,r]

        if s == 0:
            x = img_data
        else:
            x = x + img_data

    y = x / nsubj
    b = reference_img_data > y
    B = B + b.astype('int16')
    if args.log_and_write_permutations or args.log_permutations:
        if args.mask:
            P[:,p] = y
        else:
            P[:,:,:,p] = y

end = timer()
print("Permutation time: {seconds:.4f} seconds".format(seconds=end-middle))
if args.mask:
    B3D = np.zeros(reference_img.shape)
    B3D[mask] = B
    B_img = nib.Nifti1Image(B3D, affine=reference_img.affine, header=reference_img.header)
    B3D = None
else:
    B_img = nib.Nifti1Image(B, reference_img.affine)
nib.save(B_img, args.output_filename)

# Clear variables we won't need anymore
B_img = None
hdr = None
dat = None

middle = timer()
bar = progressbar.ProgressBar()
clustersizes = [[] for p in range(len(args.pthreshold))]
#for p in bar(range(nperm)):
# HACK!!!!
nn = 1000 if 1000 < nperm else nperm
for p in bar(range(nn)):
    B = np.zeros(reference_img_data.shape, dtype='int16')
    if args.mask:
        b = P[:,p,None] > P
        B = np.sum(b,axis=1)
    else:
        b = P[:,:,:,p,None] > P
        B = np.sum(b,axis=3)

    #for i in range(nperm):
    #    b = P[:,:,:,p] > P[:,:,:,i]
    #    B = B + b.astype('int16')

    # B contains, for each voxel, the number of times the reference value was
    # larger than values in the permutation set. Here, we threshold this image
    # so that if a value was larger some proportion of the time the value
    # is set to 1, otherwise it is set to 0.
    for j,pthreshold in enumerate(args.pthreshold):
        t = np.array(B > (nperm * (1-pthreshold)), dtype='int16')
        # The previous step may have left clusters of contiguous 1s in the 3D
        # volume. The following function will label each point in the volume, so
        # that points that share a label belong to the same cluster. Each cluster
        # has a unique label.
        if args.mask:
            Buffer3D = np.zeros(reference_img.shape, dtype='int16')
            Buffer3D[mask] = t
            scipy.ndimage.label(Buffer3D,output=Buffer3D)
            x = np.bincount(Buffer3D.flatten())
        else:
            scipy.ndimage.label(t,output=t)
            # For each unique label, we will tabulate how many points were assigned
            # that label. In other words, we will get the size of each labeled cluster.
            x = np.bincount(t.flatten())
        # Note two things: voxels that did not exceed threshold have a zero value,
        # and voxels that did exceed threshold but have no neighbors each have a
        # unique label (that is not shared by any other voxel). What we want to
        # know is how common clusters of different sizes are, so we can drop the
        # cluster labeled 0 and all unique labels.
        if len(x) > 1:
            clustersizes[j].append([y for y in x[1:] if y > 1])
        else:
            clustersizes[j].append([])

pclustsize = []
for i, pthreshold in enumerate(args.pthreshold):
    maxsizes = [max(x) for x in clustersizes[i] if x]
    if maxsizes:
        maxclustersize = max(maxsizes)
        clustersizeA = np.zeros((nperm, maxclustersize+1),dtype='int16')
        for j,x in enumerate(clustersizes[i]):
            y = np.bincount(x)
            clustersizeA[j,0:len(y)] = y

        pclustsize = np.sum(clustersizeA, axis=0)
        with open("clustsize_frequency_p{:.3f}.txt".format(pthreshold),'w') as f:
            for x in pclustsize:
                f.write(str(x)+"\n")
    else:
        with open("clustsize_frequency_p{:.3f}.txt".format(pthreshold),'w') as f:
            f.write("0\n")

end = timer()
print("Cluster time: {seconds:.4f} seconds".format(seconds=end-middle))

if args.log_and_write_permutations:
    print("Writing permutation array to nifti...")
    middle = timer()
    P_img = nib.Nifti1Image(P, reference_img.affine)
    nib.save(P_img, 'permutation_log.nii')
    end = timer()
    print("Write time: {seconds:.4f} seconds".format(seconds=end-middle))

end = timer()
print("Total time: {seconds:.4f} seconds".format(seconds=end-start))
