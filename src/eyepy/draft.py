# filepath = out_dir + 'myfirst_annotation_volume.bin' #oct binimages
# bin_reader = BINReader(filepath, header_size=0, num_slices=128, width=650, height=512, num_channels=1)

# # Reading data in the .bin format
# oct_volume = bin_reader.read_data()

# # Create an eyepy volume from the data => loading data in the .eye format
# eye_volume = eyepy.EyeVolume(data=oct_volume)
# eye_volume[64].plot(layers=True, dpi=300, output_dir=out_dir, watermark=False) # for showing each B-Scan 
# # eye_volume.run_interpolator(kind='linear')

# # Saving in a .eye file 
# eye_volume.save('/home/zeiss/mounted/Projects/Annotations/Ali/data/struc_1536x2048x2045x2_3551.eye')  # save volume as a set of sequential images, fds_testing_[1...N].png

# eye_volume[64].plot(layers=True, dpi=300, output_dir=out_dir, watermark=False) # for showing each B-Scan 
# eye_volume.save(out_dir+'bin_volume.avi')  # save volume as a movie
