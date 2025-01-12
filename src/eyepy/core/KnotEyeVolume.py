from __future__ import annotations
import logging
from scipy.interpolate import interp1d, Rbf,CubicSpline
import numpy as np
from eyepy.core.eyevolume import EyeVolume

logger = logging.getLogger('eyepy.core.knoteyevolume')

# These are 3 functions for creating knot_points
# 1 
def calculate_tangent(point1, point2):
    # Calculate the tangent vector between two points.
    return [(point2[0] - point1[0]), (point2[1] - point1[1])]

# 2 
def normalize_vector(vector):
    # Normalize a vector to a desired length.
    length = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
    if length > 0:
        return [vector[0] / length, vector[1] / length]
    else:
        return vector

# 3
def estimate_control_points(knot_points, cp_distance):
    control_points = []

    # Calculate control points for the first knot
    knot0 = knot_points[0]
    knot1 = knot_points[1]
    tangent_out = calculate_tangent(knot0['knot_pos'], knot1['knot_pos'])
    tangent_out = normalize_vector(tangent_out)
    cp_in_pos = [knot0['knot_pos'][0] + cp_distance * tangent_out[0], knot0['knot_pos'][1] + cp_distance * tangent_out[1]]
    control_points.append({
        'knot_pos': knot0['knot_pos'],
        'cp_in_pos': cp_in_pos,
        'cp_out_pos': knot0['knot_pos'],  # Control point out is the same as the knot for the first point
    })

    # Calculate control points for the middle knots
    for i in range(1, len(knot_points) - 1):
        knot = knot_points[i]
        prev_knot = knot_points[i - 1]
        next_knot = knot_points[i + 1]

        tangent_in = calculate_tangent(prev_knot['knot_pos'], knot['knot_pos'])
        tangent_out = calculate_tangent(knot['knot_pos'], next_knot['knot_pos'])

        tangent_in = normalize_vector(tangent_in)
        tangent_out = normalize_vector(tangent_out)

        cp_in_pos = [knot['knot_pos'][0] - cp_distance * tangent_in[0], knot['knot_pos'][1] - cp_distance * tangent_in[1]]
        cp_out_pos = [knot['knot_pos'][0] + cp_distance * tangent_out[0], knot['knot_pos'][1] + cp_distance * tangent_out[1]]

        control_points.append({
            'knot_pos': knot['knot_pos'],
            'cp_in_pos': cp_in_pos,
            'cp_out_pos': cp_out_pos,
        })

    # Calculate control points for the last knot
    knot_last = knot_points[-1]
    knot_second_last = knot_points[-2]
    tangent_in = calculate_tangent(knot_second_last['knot_pos'], knot_last['knot_pos'])
    tangent_in = normalize_vector(tangent_in)
    cp_out_pos = [knot_last['knot_pos'][0] - cp_distance * tangent_in[0], knot_last['knot_pos'][1] - cp_distance * tangent_in[1]]
    control_points.append({
        'knot_pos': knot_last['knot_pos'],
        'cp_in_pos': knot_last['knot_pos'],  # Control point in is the same as the knot for the last point
        'cp_out_pos': cp_out_pos,
    })

    return control_points


# a self-defined class with its parent featrues beside having new methods
class KnotEyeVolume(EyeVolume):
    
    def __reduce__(self):
        # The __reduce__ method should return a tuple containing two elements:
        # 1. A callable that will be used to reconstruct the object.
        # 2. A tuple of arguments to pass to the callable.
    
        # In this example, we use the class itself as the callable (constructor),
        # and pass the data attribute as an argument to recreate the object.
        return (self.__class__, (self.data,))
    
    # my self-defined function for doing the interpolation 
    def run_interpolator(self, kind='linear'):
               
        cmp_interpolated_slices = []
        for key in list(self.layers.keys()):
            label_volume = self.layers[key].data            

            # Find the indices of non-NaN rows
            non_nan_indices = ~np.isnan(label_volume).any(axis=1)
            # Find the indices of NaN rows
            nan_indices = np.isnan(label_volume).any(axis=1)            
            # Convert the boolean arrays to integer indices
            labeled_indices = np.where(non_nan_indices)[0]
            unlabeled_indices = np.where(nan_indices)[0]

            
            interpolated_volume = np.zeros_like(label_volume)
                
            # Copy the labeled slices to our newly interpolated array
            interpolated_volume[labeled_indices] = label_volume[labeled_indices] #    
    
            for i in range(0, len(unlabeled_indices) - 1, 2):
                idx1 = labeled_indices[labeled_indices < unlabeled_indices[i]][-1] # every third slice, for instance, the first ...
                idx2 = labeled_indices[labeled_indices > unlabeled_indices[i+1]][0] # and the fourth 
                adslice1 = label_volume[idx1]
                adslice2 = label_volume[idx2]
                
                if kind == 'linear':
                    f = interp1d([idx1,idx2], [adslice1,adslice2],axis=0,kind='linear') #                
                    xnew = f([unlabeled_indices[i],unlabeled_indices[i+1]])
                        
                elif kind == 'cubic':
                    f = CubicSpline([idx1,idx2], [adslice1,adslice2])
                    xnew = f([unlabeled_indices[i],unlabeled_indices[i+1]])
                    
                elif kind == 'rbf':
                    f = Rbf([idx1,idx2], [adslice1,adslice2]) # thin_plate , 0.4 0.1 => 10
                    xnew = f([unlabeled_indices[i],unlabeled_indices[i+1]])
                    
                interpolated_volume[unlabeled_indices[i]] = xnew[0] 
                interpolated_volume[unlabeled_indices[i+1]] = xnew[1] 
                cmp_interpolated_slices.append(unlabeled_indices[i])
                cmp_interpolated_slices.append(unlabeled_indices[i+1])
            
            # Check for zero values along each row and create a boolean mask
            has_zero_mask = np.any(interpolated_volume == 0, axis=1)
            # Count the number of rows with at least one zero
            num_records_with_zero = np.sum(has_zero_mask)
            if num_records_with_zero == 1: 
                indx = list(set(unlabeled_indices) - set(cmp_interpolated_slices))
                interpolated_volume[indx[0]] = interpolated_volume[indx[0]-1]
            
            elif num_records_with_zero == 2:                       
                indx = list(set(unlabeled_indices) - set(cmp_interpolated_slices))
                interpolated_volume[indx[0]] = interpolated_volume[indx[0]-1]
                interpolated_volume[indx[1]] = interpolated_volume[indx[0]-2]
            
            self.layers[key].data = interpolated_volume
        return self

    def set_knotpoints(self, dnots=10, cp_distance=10.0):
        for key in list(self.layers.keys()):
            label_volume = self.layers[key].data      
            myknot_points_all = []
    
            for i in range(0, len(label_volume)):
                myknot_points = []
                start_p = 0 + 0.5
                myknot_points.append({'knot_pos': [start_p, label_volume[i][0]]})
                
                end_p = label_volume.shape[1] - 0.5
                myknot_points.append({'knot_pos': [end_p, label_volume[i][-1]]})
    
                for j in range(int(start_p) + dnots, int(end_p), dnots):
                    myknot_points.append({'knot_pos': [j + 0.5, label_volume[i][j]]})
    
                myknot_points = sorted(myknot_points, key=lambda x: x['knot_pos'][0])
                myknot_points_all.append(estimate_control_points(myknot_points, cp_distance))
            
            myknot_points_all = {f'{idx}': lst for idx, lst in enumerate(myknot_points_all)}
            self.layers[key].knots.clear()
            self.layers[key].knots.update(myknot_points_all)
        
        return self
    

    