from keras_retinanet.utils.anchors import anchor_targets_bbox
from ..preprocessing.generator import Generator
import numpy as np

def target4joints(anchors, regression, annotation, std=0.2, mean=0):
    """Convert bbox regression to a bbox
    regression for a set of joints.

    """
    # 1. Get ground-truth
    gt_joints = annotation["vertices"].flatten()
    n_vertices = gt_joints.shape[-1]//2
    gt_bbox = annotation["bboxes"][0,:-1]
    
    # 2. Get the distance between the joints and their anchors
    bbox_rp = np.tile(gt_bbox, (1,n_vertices//2))
    delta_joints_2_bboxes = bbox_rp - gt_joints
    
    # 2b. Copy delta for each anchor
    num_anchors = anchors.shape[0]
    delta_joints_2_bboxes = np.tile(delta_joints_2_bboxes, (num_anchors,1))
    
    # 3. Get the anchors width/height for transformation
    anchor_widths  = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    
    # 4. Retransform back the regression
    tr_regress = regression[:,:-1] * std + mean
    tr_regress *= np.tile(np.array([anchor_widths, anchor_heights]).T, (1,2))
    
    # Remove  delta to tr_regress
    tr_regress = np.tile(tr_regress, (1,n_vertices//2)) - delta_joints_2_bboxes
    
    # Retransform back into normalized coordinate
    tr_regress /= np.tile(np.array([anchor_widths, anchor_heights]).T, (1,n_vertices))
    tr_regress =  (tr_regress-mean)/std
        
    # Add label row to regression
    new_regression = np.zeros((num_anchors, tr_regress.shape[-1]+1))
    new_regression[:,:-1] = tr_regress
    new_regression[:,-1] = regression[:,-1]
    
    return new_regression


    
def compute_anchor_targets_from_landmarks(anchors, 
                                          image_group, 
                                          annotations_group, 
                                          num_classes, 
                                          negative_overlap=0.4,
                                        positive_overlap=0.5):
    """annotations_group: List of annotations (np.array of shape (N, 5) for (x1, y1, x2, y2, label))."""
    regression , labels = anchor_targets_bbox(anchors, image_group, annotations_group, num_classes, negative_overlap, positive_overlap )
    
    batch_size = len(annotations_group)
    n_vertices = annotations_group[0]["vertices"].flatten().shape[-1]//2
    regress_group = np.zeros((batch_size, regression.shape[1], int(2*n_vertices + 1) ))
    
    for i in range(batch_size):
        new_regression = target4joints(anchors, regression[i], annotations_group[i])
        regress_group[i,:,:] = new_regression.copy()
    
    return regress_group, labels



class PolygonGenerator(Generator):
    
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        super(PolygonGenerator, self).__init__(compute_anchor_targets=compute_anchor_targets_from_landmarks, **kwargs)
     
    def size(self):
        return len(self.dataset)
    
    def num_classes(self):
        return 1
    
    def has_label(self, label):
        return label==0
    
    def has_name(self, name):
        return name=="polygon"

    def name_to_label(self, name):
        return 0

    def label_to_name(self, label):
        return "polygon"

    def image_aspect_ratio(self, image_index):
        return 1
    
    def joints2bboxes(self, joints):
        """Convert a set of vertices : (n_vertices,2) to the maximum bounding box"""
        annots = np.zeros((1, 5)) # 5-> 4+1
        annots[:,0] = np.min(joints[:,0], axis=0)
        annots[:,1] = np.min(joints[:,1], axis=0)
        annots[:,2] = np.max(joints[:,0], axis=0)
        annots[:,3] = np.max(joints[:,1], axis=0)
        return annots
    
    def load_image(self, image_index):
        return self.dataset.load_image(image_index)
    
    
    def load_annotations(self, image_index):
        joints = self.dataset.load_joints(image_index, normalized=False)
        bboxes = self.joints2bboxes(joints)
        
        annotations = {'labels': np.zeros((1,)), 
                       'vertices': joints,
                       'bboxes': bboxes}
        return annotations
    
    def filter_annotations(self, image_group, annotations_group, group):
        """ Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        return image_group, annotations_group
    
    def random_transform_group_entry(self, image, annotations, transform=None):
        return image, annotations
    
    def random_transform_group(self, image_group, annotations_group):
        return image_group, annotations_group
    
    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)
        return inputs, targets
    
    
    def get_anchors(self):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        group = [0]
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # compute network inputs
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        return self.generate_anchors(max_shape)

