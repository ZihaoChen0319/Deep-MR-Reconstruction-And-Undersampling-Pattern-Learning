import numpy as np
import SimpleITK as sitk
from functools import wraps


def slicing_decorator(func):
    """
    Based on https://discourse.itk.org/t/resampleimagefilter-4d-images/2172/3

    A function decorator which extracts image slices of N-1 dimensions and calls func on each slice. The resulting
     images are then concatenated together with JoinSeries.

    :param func: A function which take a SimpleITK Image as it's first argument
    :return: The result of running func on each slice of image.
    """

    @wraps(func)
    def slice_by_slice(image, *args, **kwargs):
        size = list(image.GetSize())
        if len(size) == 4:
            number_of_slices = size[-1]
            extract_size = size
            extract_index = [0] * image.GetDimension()

            img_list = []

            extract_size[-1] = 0
            extractor = sitk.ExtractImageFilter()
            extractor.SetSize(extract_size)

            for slice_idx in range(0, number_of_slices):
                extract_index[-1] = slice_idx
                extractor.SetIndex(extract_index)

                img_list.append(func(extractor.Execute(image), *args, **kwargs))

            return sitk.JoinSeries(img_list, image.GetOrigin()[-1], image.GetSpacing()[-1])
        else:
            return func(image, *args, **kwargs)

    return slice_by_slice


@slicing_decorator
def sitk_resample(itk_image, target_spacing, is_label=False):
    """
    Based on https://gist.github.com/mrajchl/ccbd5ed12eb68e0c1afc5da116af614a

    """
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / target_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

