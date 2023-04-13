import os
import math
import cv2
import numpy as np
import mahotas.polygon as ploygon_to_mask
from lxml import etree


def check_policy(target_str, mask_policy_dict):
    in_policy = False
    mask_key = None
    for policy_key in mask_policy_dict.keys():
        in_policy = policy_key in target_str
        if in_policy is True:
            mask_key = policy_key
            break
    return in_policy, mask_key


def get_mask_image(image_shape, mask_xml_path, level, downsizing_per_level=2, mask_policy_dict={'None': 255}):
    image_shape = np.array(image_shape) // (downsizing_per_level ** level)

    mask_array = np.zeros(image_shape, dtype=np.uint8)
    if os.path.exists(mask_xml_path) is False:
        print(f"{mask_xml_path} not exist.")
        return mask_array
    mask_xml_tree = etree.parse(mask_xml_path)
    ASAP_Annotations_tree = mask_xml_tree.getroot()
    Annotations_tree = ASAP_Annotations_tree[0]

    for Annotation_tree in Annotations_tree:
        Annotaion_Type = Annotation_tree.attrib['Type']
        Annotation_PartOfGroup = Annotation_tree.attrib['PartOfGroup']
        Coordinates_tree = Annotation_tree[0]
        is_in_policy, mask_key = check_policy(Annotation_PartOfGroup,
                                              mask_policy_dict)
        if (Annotaion_Type == "Spline") and (is_in_policy == True):
            mask_value = mask_policy_dict[mask_key]
            polygon_points = []
            for Coordinate in Coordinates_tree.iter("Coordinate"):
                x = float(Coordinate.attrib["X"])
                y = float(Coordinate.attrib["Y"])
                polygon_points.append((round(
                    y) // (downsizing_per_level ** level), round(x) // (downsizing_per_level ** level)))
            ploygon_to_mask.fill_polygon(
                polygon_points, mask_array, mask_value)

    return mask_array


def get_rec_info_list(xml_path, patch_size):

    Rec_info_list = []
    mask_xml_tree = etree.parse(xml_path)
    ASAP_Annotations_tree = mask_xml_tree.getroot()
    Annotations_tree = ASAP_Annotations_tree[0]

    for Annotation_tree in Annotations_tree:
        Annotation_Name = Annotation_tree.attrib["Name"].replace("\n", "")
        Annotaion_Type = Annotation_tree.attrib['Type']
        Annotation_PartOfGroup = Annotation_tree.attrib['PartOfGroup']
        Coordinates_tree = Annotation_tree[0]

        if Annotaion_Type == "Rectangle" and Annotation_PartOfGroup != "lvi":
            x_list, y_list = [], []
            for Coordinate in Coordinates_tree.iter("Coordinate"):
                x = float(Coordinate.attrib["X"])
                y = float(Coordinate.attrib["Y"])
                x_list.append(x), y_list.append(y)
            row_min, row_max = math.floor(min(y_list)), math.ceil(max(y_list))
            col_min, col_max = math.floor(min(x_list)), math.ceil(max(x_list))
            row_quotient, row_mod = (
                row_max - row_min) // patch_size, (row_max - row_min) % patch_size
            row_padding = math.ceil((patch_size - row_mod) / 2)
            col_quotient, col_mod = (
                col_max - col_min) // patch_size, (col_max - col_min) % patch_size
            col_padding = math.ceil((patch_size - col_mod) / 2)
            row_idx_list = [row_min + patch_size * row_idx -
                            row_padding for row_idx in range(0, row_quotient + 1)]
            col_idx_list = [col_min + patch_size * col_idx -
                            col_padding for col_idx in range(0, col_quotient + 1)]
            Rec_info = [Annotation_Name, row_idx_list, col_idx_list]
            Rec_info_list.append(Rec_info)
    return Rec_info_list


def remove_orange_peel(mask_array, mask_policy_dict, remove_region_ratio=0.01):
    mask_area = np.prod(mask_array.shape[:2])
    for key, value in mask_policy_dict.items():
        mask_region = np.prod(mask_array == value, axis=-1).astype("uint8")
        if np.sum(mask_region) == 0:
            continue
        else:
            mask_num, mask_region = cv2.connectedComponents(mask_region)
            for mask_index in range(1, mask_num):
                mask_index_region = mask_region == mask_index
                mask_boundary_sum = np.sum(mask_index_region[0, :]) + np.sum(mask_index_region[-1, :]) + \
                    np.sum(mask_index_region[:, 0]) + \
                    np.sum(mask_index_region[:, -1])
                if mask_boundary_sum > 0:
                    region_ratio = np.sum(mask_index_region) / mask_area
                    if region_ratio <= remove_region_ratio:
                        mask_array[mask_index_region] = 0

    return mask_array


def resize_with_preserve_rgb_value(mask_array, target_dsize, mask_policy_dict):
    x, y = target_dsize
    resize_mask_array = np.zeros((y, x, 3), dtype=mask_array.dtype)
    for key, value in mask_policy_dict.items():
        mask_region = np.prod(mask_array == value,
                              axis=-1).astype("uint8") * 255
        if np.sum(mask_region) == 0:
            continue
        else:
            mask_region = cv2.resize(mask_region, (x, y), cv2.INTER_LINEAR)
            resize_mask_array[mask_region > 127.5] = value
    return resize_mask_array
