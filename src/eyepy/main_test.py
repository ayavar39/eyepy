'''
This script is used for working with eyepy, oct converter and eyelab
in this script, we have tested some functionality of the mentioned libraries
This way, it is called main_test

'''
from oct_converter.readers.utilfs import load_eye_from_bin, save_bin2eye, save_eye2bin, save_eye2hdf5, load_eye_from_hdf5
import eyepy

import warnings
import time
# Set the warning filter to "ignore" (ignore all warnings)
warnings.filterwarnings("ignore")

start = time.time()
out_dir = 'D:/dataset/eyepy/'

# 1- Loading, creating and returting a .eye object  from a .bin path 
eye_volume1 = load_eye_from_bin('D:/dataset/heiderlberg_oct1_.bin', header_size=0, num_slices=128, width=650, height=512, num_channels=1, prototype='other')
# eye_volume1[64].plot()

# 2- Saving a .bin file in a .eye file 
save_bin2eye('D:/dataset/heiderlberg_oct1_.bin', 'D:/dataset/heiderlberg_oct1_.eye', header_size=0, num_slices=128, width=650, height=512, num_channels=1, prototype='other')

# 3- Saving a .eye file content to a .bin file
save_eye2bin('D:/dataset/heiderlberg_oct1_.eye','D:/dataset/heiderlberg_oct2.bin')

# 
eye_volume2 = load_eye_from_bin('D:/dataset/heiderlberg_oct2_structure.bin', header_size=0, num_slices=128, width=650, height=512, num_channels=1, prototype='other')


# 4- Saving a .eye object to a .hdf5 file
save_eye2hdf5('D:/dataset/heiderlberg_oct1_.eye', 'D:/dataset/heiderlberg_oct2.hdf5')


# 5- Loading a .eye object from a .hdf5 file
eye_vol2 = load_eye_from_hdf5('D:/dataset/heiderlberg_oct2.hdf5')


# Loading .eye file 
eyefile_path = 'D:/dataset/temp_annotated_heiderlberg.eye'
eyefile_path1 = 'D:/dataset/temp_annotated_heiderlberg1.eye'
annotated_eye_volume = eyepy.EyeVolume.load(eyefile_path)

annotated_eye_volume2 = eyepy.KnotEyeVolume.load(eyefile_path)

# annotated_eye_volume[3].plot(layers=True,dpi=300, output_dir=out_dir)
# annotated_eye_volume.layers

temp = annotated_eye_volume2 
temp = temp.run_interpolator(kind='cubic')

temp = temp.set_knotpoints(dnots=40)
temp.save(eyefile_path1)

# Save the eye file
# eye_volume.save(eye_path)

# eye_volume = eyepy.EyeVolume.load(out_dir + 'annotated_heiderlberg_oct1_.eye')
# # eye_volume[63].plot(layers=True, dpi=300, output_dir=out_dir, watermark=False) # for showing each B-Scan 

# eye_volume.run_interpolator(kind='linear')
# eye_volume[63].plot(layers=True, dpi=300, watermark=False) # for showing each B-Scan 

# eye_volume.save(out_dir + 'interpolated_annotated_heiderlberg_oct1_.eye')

# end = time.time()
# print(end - start)


