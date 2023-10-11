from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.paths import nnUNet_results
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import nibabel as nib
from nibabel import io_orientation
import tempfile
from scipy import ndimage
import psutil
from joblib import Parallel, delayed
import importlib
import numpy as np
import time
import os
import gc

cupy_available = importlib.util.find_spec("cupy") is not None
cucim_available = importlib.util.find_spec("cucim") is not None

gc.collect()

labels = {
    "bones": 1,
    "arteries": 2,
    "veins": 3,
    "muscles": 4,
    "spleen": 5,
    "kidney_right": 6,
    "kidney_left": 7,
    "gallbladder": 8,
    "liver": 9,
    "liver_vessels": 10,
    "stomach": 11,
    "pancreas": 12,
    "adrenal_gland_right": 13,
    "adrenal_gland_left": 14,
    "esophagus": 15,
    "brain": 16,
    "small_bowel": 17,
    "duodenum": 18,
    "colon": 19,
    "urinary_bladder": 20,
    "lung_upper_lobe_left": 21,
    "lung_lower_lobe_left": 22,
    "lung_upper_lobe_right": 23,
    "lung_middle_lobe_right": 24,
    "lung_lower_lobe_right": 25,
    "lung_trachea_bronchia": 26,
    "lung_vessels": 27,
    "heart_myocardium": 28,
    "heart_atrium_left": 29,
    "heart_ventricle_left": 30,
    "heart_atrium_right": 31,
    "heart_ventricle_right": 32,
    "liver_formation": 33,
    "lung_formation": 34,
    "kidney_formation": 35,
    "pleural_effusion": 36,
    "intracerebral_hemorrhage": 37
}

def resample_img(img, zoom=0.5, order=0, nr_cpus=-1):
    """
    img: [x,y,z,(t)]
    zoom: 0.5 will halfen the image resolution (make image smaller)

    Resize numpy image array to new size.

    Faster than resample_img_nnunet.
    Resample_img_nnunet maybe slighlty better quality on CT (but not sure).
    
    Works for 2D and 3D and 4D images.
    """
    def _process_gradient(grad_idx):
        return ndimage.zoom(img[:, :, :, grad_idx], zoom, order=order)

    dim = len(img.shape)

    # Add dimesions to make each input 4D
    if dim == 2: 
        img = img[..., None, None]
    if dim == 3: 
        img = img[..., None]

    nr_cpus = psutil.cpu_count() if nr_cpus == -1 else nr_cpus
    img_sm = Parallel(n_jobs=nr_cpus)(delayed(_process_gradient)(grad_idx) for grad_idx in range(img.shape[3]))
    img_sm = np.array(img_sm).transpose(1, 2, 3, 0)  # grads channel was in front -> put to back
    # Remove added dimensions
    # img_sm = img_sm[:,:,:,0] if img_sm.shape[3] == 1 else img_sm  # remove channel dim if only 1 element
    if dim == 3:
        img_sm = img_sm[:,:,:,0]
    if dim == 2:
        img_sm = img_sm[:,:,0,0]
    return img_sm

def resample_img_cucim(img, zoom=0.5, order=0, nr_cpus=-1):
    """
    Completely speedup of resampling compare to non-gpu version not as big, because much time is lost in 
    loading the file and then in copying to the GPU.

    For small image no significant speedup.
    For large images reducing resampling time by over 50%.

    On our slurm gpu cluster it is actually slower with cucim than without it.
    """
    import cupy as cp
    from cucim.skimage.transform import resize

    img = cp.asarray(img)  # slow
    new_shape = (np.array(img.shape) * zoom).round().astype(np.int32)
    resampled_img = resize(img, output_shape=new_shape, order=order, mode="edge", anti_aliasing=False)  # very fast
    resampled_img = cp.asnumpy(resampled_img)  # Alternative: img_arr = cp.float32(resampled_img.get())   # very fast
    return resampled_img

def resample_img_nnunet(data, mask=None, original_spacing=1.0, target_spacing=2.0):
    """
    Args:
        data: [x,y,z]
        mask: [x,y,z]
        original_spacing:
        target_spacing:

    Zoom = original_spacing / target_spacing
    (1 / 2 will reduce size by 50%)

    Returns:
        [x,y,z], [x,y,z]
    """
    from .resample_nnunet import resample_patient
    
    if type(original_spacing) is float:
        original_spacing = [original_spacing,] * 3
    original_spacing = np.array(original_spacing)

    if type(target_spacing) is float:
        target_spacing = [target_spacing,] * 3
    target_spacing = np.array(target_spacing)

    data = data.transpose((2, 0, 1))  # z is in front for nnUnet
    data = data[None, ...]  # [1,z,x,y], nnunet requires a channel dimension
    if mask is not None:
        mask = mask.transpose((2, 0, 1))
        mask = mask[None, ...]

    def move_last_elem_to_front(l):
        return np.array([l[2], l[0], l[1]])

    # if anisotropy too big, then will resample z axis separately with order=0
    original_spacing = move_last_elem_to_front(original_spacing)
    target_spacing = move_last_elem_to_front(target_spacing)
    data_res, mask_res = resample_patient(data, mask, original_spacing, target_spacing, force_separate_z=None)

    data_res = data_res[0,...] # remove channel dimension
    data_res = data_res.transpose((1, 2, 0)) # Move z to back
    if mask is not None:
        mask_res = mask_res[0,...]
        mask_res = mask_res.transpose((1, 2, 0))
    return data_res, mask_res

def change_spacing(img_in, new_spacing=1.25, target_shape=None, order=0, nr_cpus=1,
                   nnunet_resample=False, dtype=None, remove_negative=False, force_affine=None):
    """
    Resample nifti image to the new spacing (uses resample_img() internally).
    
    img_in: nifti image
    new_spacing: float or sequence of float
    target_shape: sequence of int (optional)
    order: resample order (optional)
    nnunet_resample: nnunet resampling will use order=0 sampling for z if very anisotropic. Sometimes results 
                     in a little bit less blurry results
    dtype: output datatype
    remove_negative: set all negative values to 0. Useful if resampling introduced negative values.
    force_affine: if you pass an affine then this will be used for the output image (useful if you have to make sure
                  that the resampled has identical affine to some other image. In this case also set target_shape.)

    Works for 2D and 3D and 4D images.

    If downsampling an image and then upsampling again to original resolution the resulting image can have
    a shape which is +-1 compared to original shape, because of rounding of the shape to int.
    To avoid this the exact output shape can be provided. Then new_spacing will be ignored and the exact
    spacing will be calculated which is needed to get to target_shape.
    In this case however the calculated spacing can be slighlty different from the desired new_spacing. This will
    result in a slightly different affine. To avoid this the desired affine can be writen by force with "force_affine".

    Note: Only works properly if affine is all 0 except for diagonal and offset (=no rotation and sheering)
    """
    data = img_in.get_fdata()  # quite slow
    old_shape = np.array(data.shape)
    img_spacing = np.array(img_in.header.get_zooms())

    if len(img_spacing) == 4:
        img_spacing = img_spacing[:3]  # for 4D images only use spacing of first 3 dims

    if type(new_spacing) is float:
        new_spacing = [new_spacing,] * 3   # for 3D and 4D
    new_spacing = np.array(new_spacing)

    if len(old_shape) == 2:
        img_spacing = np.array(list(img_spacing) + [new_spacing[2],])

    if target_shape is not None:
        # Find the right zoom to exactly reach the target_shape.
        # We also have to adapt the spacing to this new zoom.
        zoom = np.array(target_shape) / old_shape  
        new_spacing = img_spacing / zoom  
    else:
        zoom = img_spacing / new_spacing

    # copy very important; otherwise new_affine changes will also be in old affine
    new_affine = np.copy(img_in.affine)

    # This is only correct if all off-diagonal elements are 0
    # new_affine[0, 0] = new_spacing[0] if img_in.affine[0, 0] > 0 else -new_spacing[0]
    # new_affine[1, 1] = new_spacing[1] if img_in.affine[1, 1] > 0 else -new_spacing[1]
    # new_affine[2, 2] = new_spacing[2] if img_in.affine[2, 2] > 0 else -new_spacing[2]

    # This is the proper solution
    # Scale each column vector by the zoom of this dimension
    new_affine = np.copy(img_in.affine)
    new_affine[:3, 0] = new_affine[:3, 0] / zoom[0]
    new_affine[:3, 1] = new_affine[:3, 1] / zoom[1]
    new_affine[:3, 2] = new_affine[:3, 2] / zoom[2]

    # Just for information: How to get spacing from affine with rotation:
    # Calc length of each column vector:
    # vecs = affine[:3, :3]
    # spacing = tuple(np.sqrt(np.sum(vecs ** 2, axis=0)))

    if nnunet_resample:
        new_data, _ = resample_img_nnunet(data, None, img_spacing, new_spacing)
    else:
        if cupy_available and cucim_available:
            new_data = resample_img_cucim(data, zoom=zoom, order=order, nr_cpus=nr_cpus)  # gpu resampling
        else:
            new_data = resample_img(data, zoom=zoom, order=order, nr_cpus=nr_cpus)  # cpu resampling
        
    if remove_negative:
        new_data[new_data < 1e-4] = 0

    if dtype is not None:
        new_data = new_data.astype(dtype)

    if force_affine is not None:
        new_affine = force_affine

    return nib.Nifti1Image(new_data, new_affine)

def load_and_reorient(data, reorent=(0, 1, 2)):
    non_reorent_img = nib.load(data)
    assert len(non_reorent_img.shape) == 3, 'only 3d images are supported by NibabelIO'
    original_affine = non_reorent_img.affine
    reoriented_image = non_reorent_img.as_reoriented(io_orientation(original_affine))
    reoriented_affine = reoriented_image.affine
    reorent_data = reoriented_image.get_fdata().transpose(reorent)

    return non_reorent_img, reoriented_image, original_affine, reoriented_affine, reorent_data

def check_if_shape_and_affine_identical(img_1, img_2):
    
    if not np.array_equal(img_1.affine, img_2.affine):
        print("Affine in:")
        print(img_1.affine)
        print("Affine out:")
        print(img_2.affine)
        print("Diff:")
        print(np.abs(img_1.affine-img_2.affine))
        print("WARNING: Output affine not equal to input affine. This should not happen.")

    if img_1.shape != img_2.shape:
        print("Shape in:")
        print(img_1.shape)
        print("Shape out:")
        print(img_2.shape)
        print("WARNING: Output shape not equal to input shape. This should not happen.")

def predict_image(input_file, output, model, separate_masks=False ,nnunet_verbose=True, use_gpu=True):
    '''
    param: input_file - nii/nii.gz file to predict
    param: output - the output distanation  (path/to/save/)
    param: model - the directory of nnunet trained model
    '''

    assert input_file.endswith('.nii') or input_file.endswith('.nii.gz'), f"Input file format error. Input file should endswith .nii or .nii.gz. Input file: {input_file}"
    #assert output.endswith('.nii') or output.endswith('.nii.gz'), f"Output file format error. Output file should endswith .nii or .nii.gz. Output file: {output}"

    pt = time.time()

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_gpu=use_gpu,
        device=torch.device('cuda', 0),
        verbose=nnunet_verbose,
        verbose_preprocessing=True,
        allow_tqdm=True
    )

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, model),
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )

    img_in = nib.load(input_file)
    resample = 1.25
    nr_threads_resampling = 1

    with tempfile.TemporaryDirectory(prefix="mosaic_tmp_") as tmp_folder:
        # Resample image
        print(f"Resampling...")
        st = time.time()
        img_in_shape = img_in.shape
        img_in_zooms = img_in.header.get_zooms()
        print(f'Original spacing: {img_in_zooms}')
        img_in_rsp = change_spacing(img_in, [resample, resample, resample],
                                    order=3, dtype=np.int32, nr_cpus=nr_threads_resampling)  # 4 cpus instead of 1 makes it a bit slower
        print(f"from shape {img_in_shape} to shape {img_in_rsp.shape}")
        print(f"Resampled in {time.time() - st:.2f}s")

        nib.save(img_in_rsp, f'{str(tmp_folder)}/resampled.nii.gz')

        # Load resampled image with NibabelIOWithReorient
        img, props = NibabelIOWithReorient().read_images([f"{str(tmp_folder)}/resampled.nii.gz"])
        print(f"\nLoaded resampled image with shape {img[0].shape} and spacing: {props['spacing']}")

        # Predict
        print(f"\nPredicting...")
        st = time.time()
        pred = predictor.predict_single_npy_array(img, props, None, None, False)
        print(f"Predicted in {time.time() - st:.2f}s")

        # Save reorient mask
        NibabelIOWithReorient().write_seg(seg=pred,
                  output_fname=f'{str(tmp_folder)}/reorient_pred.nii.gz',
                  properties=props)

        pred_reorient_img = nib.load(f'{str(tmp_folder)}/reorient_pred.nii.gz')
        pred_reorient_data = pred_reorient_img.get_fdata()

        # Resample to original spacing
        print(f"\nResampling to original spacing...")
        st = time.time()
        mask_in_rsp = change_spacing(pred_reorient_img, [resample, resample, resample], img_in_shape,
                                        order=0, dtype=np.uint8, nr_cpus=nr_threads_resampling,
                                        force_affine=img_in.affine)
        print(f"from shape {pred_reorient_data.shape} to shape {mask_in_rsp.shape}")
        print(f"Resampled in {time.time() - st:.2f}s\n")

        check_if_shape_and_affine_identical(img_in, mask_in_rsp)

        if separate_masks:
            if not os.path.exists(os.path.join(output, 'segmentations')):
                os.mkdir(os.path.join(output, 'segmentations'))
            for label, indx in labels.items():
                label_mask = np.array(mask_in_rsp.get_fdata() == indx).astype(np.uint8)
                nib.save(nib.Nifti1Image(label_mask, img_in.affine), os.path.join(output, 'segmentations', label+'.nii.gz'))

        # Save results
        nib.save(mask_in_rsp, os.path.join(output, 'total.nii.gz'))

        del img_in, img, props, predictor, pred, mask_in_rsp
        gc.collect()

        print(f"Done in {time.time() - pt:.2f}s")