import torch 
import json 
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertTokenizer, BertModel
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage
from collections import defaultdict
from PIL import Image, ImageDraw
from copy import deepcopy
import os 
import torchvision.transforms as transforms
import torchvision
from .base_dataset import BaseDataset, check_filenames_in_zipdata, recalculate_box_and_verify_if_valid  
from io import BytesIO
import random

from .tsv import TSVFile

from io import BytesIO
import base64
from PIL import Image
import numpy as np
import scipy.io as sio
import torch
from transformers import CanineTokenizer, CanineModel
import cv2
import clip


def extract_image_file_paths(mat_contents, image_rootdir):
    image_files = []
    for image_file in mat_contents['imnames'][0]:
        image_files.append(os.path.join(image_rootdir, image_file[0]))
    return image_files


def get_words(mat_contents, index):
    words = []
    for word in mat_contents['txt'][0][index]:
        for indiv in word.split():
            for part in indiv.split('\n'):
                if part.strip() != '':
                    words.append(part.strip())
    return words

def get_bb(mat_contents, index):
    # # print(mat_contents)
    # # print("Bounding Box for Image",mat_contents['wordBB'][0][index])
    # try:
    #     reshaped_array = mat_contents['wordBB'][0][index].reshape(mat_contents['wordBB'][0][index].shape[2], 8)
    #     selected_indices = reshaped_array[:, [3, 7, 1, 5]]
    # except:
    #     print("Matcontents Shape",mat_contents['wordBB'][0][index].shape)
    #     # write an empty np array
    #     selected_indices = np.array([])

    #     pass
    # return selected_indices

        result_image = []
        shape =  mat_contents['wordBB'][0][index].shape
        b,c = 4,1
        if(len(shape) > 2):
            a, b, c = shape[0], shape[1], shape[2]
        else:
            a, b, c = shape[0], shape[1], 1


        for i in range(c):
            result_image.append([])

        for i in range(c):
            for j in range(b):
                if c==1:
                    result_image[i].append([mat_contents['wordBB'][0][index][0][j], mat_contents['wordBB'][0][index][1][j]])
                else: 
                    result_image[i].append([mat_contents['wordBB'][0][index][0][j][i], mat_contents['wordBB'][0][index][1][j][i]])


        final_indicies_actual = []
        for i in range(c):
            x0,x1,x2,x3  = result_image[i]
            actual_width, actual_height = np.abs((x0[0] - x1[0])), np.abs(x0[1] - x3[1])

            x3_actual = np.append(x0,np.array([actual_width, actual_height]))
            final_indicies_actual.append(x3_actual)
        return np.array(final_indicies_actual)

def get_embeddings(words):
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_model.to(device)

    # Initialize the final embeddings tensor
    final_embeddings = torch.zeros(len(words), bert_model.config.hidden_size)

    # Tokenize and compute embeddings for each word
    for i, word in enumerate(words):
        # Tokenize word and prepare input tensors
        token_output = bert_tokenizer(word, return_tensors='pt')
        input_ids = token_output['input_ids'].to(device)
        attention_mask = token_output['attention_mask'].to(device)

        # Get the output from the BERT model
        with torch.no_grad():
            output = bert_model(input_ids, attention_mask=attention_mask)

        # Extract the embeddings for the [CLS] token
        embeddings = output.last_hidden_state[:, 0, :]
        final_embeddings[i] = embeddings

    return final_embeddings

def get_text_description(mat_contents, index):
    words = get_words(mat_contents, index)
    image_description = mat_contents['imnames'][0][index][0]
    image_description_text = image_description.split('/')[1].split('_')[0]
    return f"this is an image of {image_description_text} which consists of the following words {','.join(words)} in the image"

def mask_image_outside_bbox(image_path, bbox):
    """
    Mask the image outside the bounding box and make the inside white.
 
    Parameters:
    - image_path: str, path to the image file.
    - bbox: tuple of (x, y, width, height), the bounding box inside which the image will be white.
 
    Returns:
    - masked_image: the image with masking applied.
    """
    # # Load the image
    # image = np.array(Image.open(image_path).convert('RGB'))

    # # Convert bbox coordinates to integers
    # x, y, width, height = map(int, bbox)

    # # Create a mask where the area inside the bbox is white
    # mask = np.zeros_like(image)
    # mask[y:y+height, x:x+width] = 255

    # # Apply the mask to the image
    # masked_image = cv2.bitwise_and(image, mask)

    # # Make the inside of the bbox white
    # masked_image[y:y+height, x:x+width] = 255

    # # Convert the masked image back to PIL format
    # masked_image_pil = Image.fromarray(masked_image)

    # return masked_image_pil
    # Load the image
    # image = np.array(Image.open(image_path).convert('RGB'))

    # Convert bbox coordinates to integers
    # x, y, width, height = map(int, bbox)

    # Create a mask where the area inside the bbox is white
    # mask = np.zeros_like(image)
    # mask[height-5:y+5, x-5:width+5] = 255

    # # Apply the mask to the image
    # masked_image = cv2.bitwise_and(image, mask)

    # masked_image_pil = Image.fromarray(masked_image)

    # # Make the inside of the bbox white
    # masked_image[y:y+height , x:x+width] = 255

    image = Image.open(image_path).convert('RGB')
    # Convert bbox coordinates to integers    
    x, y, width, height = map(int, bbox)
    
    # Calculate the bottom right coordinates
    x2 = x + width
    y2 = y + height
    
    # Crop the image using the bounding box
    cropped_image = image.crop((x, y, x2, y2))
    
    return cropped_image

def get_clip_image_embeddings(image):
    """
    Get the CLIP image embeddings for the given image. 
    Parameters:
    - image_path: str, path to the image file.
    - text: str, text to convert to embedding.
 
    Returns:
    - image_embedding: tensor, embedding of the image.
    - text_embedding: tensor, embedding of the text.
    """
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights).to(device)
    model.eval()  # Set the model to inference mode

    # Define the transformation for the input image
    preprocess = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])

    # Load and preprocess an image
    img_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor.to(device) 
    
    # Get the embedding without tracking gradients
    with torch.no_grad():
        features = model(img_tensor)[0]  # Extract features from the transformer

        # Optional: project to 768 dimensions if not already the correct size
        if features.shape[-1] != 768:
            projection = torch.nn.Linear(features.shape[-1], 768).to(device)
            projection.eval()  # Set the projection to inference mode
            embedding = projection(features)
        else:
            embedding = features

    # The output 'embedding' is a tensor which is the image representation
    #print(embedding)
    #print(embedding.shape) 
    
    return embedding
    

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    WW, HH = pil_image.size

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)

    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    # at this point, the min of pil_image side is desired image_size
    performed_scale = image_size / min(WW, HH)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    
    info = {"performed_scale":performed_scale, 'crop_y':crop_y, 'crop_x':crop_x, "WW":WW, 'HH':HH}

    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size], info


def recalculate_box_and_verify_if_valid(x, y, w, h, trans_info, image_size, min_box_size):
    """
    x,y,w,h:  the original annotation corresponding to the raw image size.
    trans_info: what resizing and cropping have been applied to the raw image 
    image_size:  what is the final image size  
    """

    x0 = x * trans_info["performed_scale"] - trans_info['crop_x'] 
    y0 = y * trans_info["performed_scale"] - trans_info['crop_y'] 
    x1 = w * trans_info["performed_scale"] - trans_info['crop_x'] 
    y1 = h * trans_info["performed_scale"] - trans_info['crop_y'] 


    # at this point, box annotation has been recalculated based on scaling and cropping
    # but some point may fall off the image_size region (e.g., negative value), thus we 
    # need to clamp them into 0-image_size. But if all points falling outsize of image 
    # region, then we will consider this is an invalid box. 
    valid, (x0, y0, x1, y1) = to_valid(x0, y0, x1, y1, image_size, min_box_size)

    if valid:
        # we also perform random flip. 
        # Here boxes are valid, and are based on image_size 
        if trans_info["performed_flip"]:
            x0, x1 = image_size-x1, image_size-x0

    return valid, (x0, y0, x1, y1)


def to_valid(x0, y0, x1, y1, image_size, min_box_size):
    valid = True

    if x0>image_size or y0>image_size or x1<0 or y1<0:
        valid = False # no way to make this box vide, it is completely cropped out 
        return valid, (None, None, None, None)

    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, image_size)
    y1 = min(y1, image_size)

    if (x1-x0)*(y1-y0) / (image_size*image_size) < min_box_size:
        valid = False
        return valid, (None, None, None, None)
     
    return valid, (x0, y0, x1, y1)

def decode_base64_to_pillow(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')

def decode_tensor_from_string(arr_str, use_tensor=True):
    arr = np.frombuffer(base64.b64decode(arr_str), dtype='float32')
    if use_tensor:
        arr = torch.from_numpy(arr)
    return arr

def decode_item(item):
    item = json.loads(item)
    item['image'] = decode_base64_to_pillow(item['image'])

    for anno in item['annos']:
        anno['image_embedding_before'] = decode_tensor_from_string(anno['image_embedding_before'])
        anno['text_embedding_before'] = decode_tensor_from_string(anno['text_embedding_before'])
        anno['image_embedding_after'] = decode_tensor_from_string(anno['image_embedding_after'])
        anno['text_embedding_after'] = decode_tensor_from_string(anno['text_embedding_after'])
    return item

def check_unique(images, fields):
    for field in fields:
        temp_list = []
        for img_info in images:
            temp_list.append(img_info[field])
        assert len(set(temp_list)) == len(temp_list), field

def clean_data(data):
    for data_info in data:
        data_info.pop("original_img_id", None)
        data_info.pop("original_id", None)
        data_info.pop("sentence_id", None)  # sentence id for each image (multiple sentences for one image)
        data_info.pop("dataset_name", None)  
        data_info.pop("data_source", None) 
        data_info["data_id"] = data_info.pop("id")


def clean_annotations(annotations):
    for anno_info in annotations:
        anno_info.pop("iscrowd", None) # I have checked that all 0 for flickr, vg, coco
        anno_info.pop("category_id", None)  # I have checked that all 1 for flickr vg. This is not always 1 for coco, but I do not think we need this annotation
        anno_info.pop("area", None)
        # anno_info.pop("id", None)
        anno_info["data_id"] = anno_info.pop("image_id")


def draw_box(img, boxes):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline ="red", width=2) # x0 y0 x1 y1 
    return img 


def xyhw2xyxy(box):
    x0, y0, w, h = box
    return [ x0, y0, x0+w, y0+h ]


def make_a_sentence(obj_names, clean=False):

    if clean:
        obj_names = [ name[:-6] if ("-other" in name) else name for name in obj_names]

    caption = ""
    tokens_positive = []
    for obj_name in obj_names:
        start_len = len(caption)
        caption += obj_name
        end_len = len(caption)
        caption += ", "
        tokens_positive.append(
            [[start_len, end_len]] # in real caption, positive tokens can be disjoint, thus using list of list
        )
    caption = caption[:-2] # remove last ", "

    return caption #, tokens_positive


def mask_for_random_drop_text_or_image_feature(masks, random_drop_embedding):
    """
    input masks tell how many valid grounding tokens for this image
    e.g., 1,1,1,1,0,0,0,0,0,0...

    If random_drop_embedding=both.  we will random drop either image or
    text feature for each token, 
    but we always make sure there is at least one feature used. 
    In other words, the following masks are not valid 
    (because for the second obj, no feature at all):
    image: 1,0,1,1,0,0,0,0,0
    text:  1,0,0,0,0,0,0,0,0

    if random_drop_embedding=image. we will random drop image feature 
    and always keep the text one.  

    """
    N = masks.shape[0]

    if random_drop_embedding=='both':
        temp_mask = torch.ones(2,N)
        for i in range(N):
            if random.uniform(0, 1) < 0.5: # else keep both features 
                idx = random.sample([0,1], 1)[0] # randomly choose to drop image or text feature 
                temp_mask[idx,i] = 0 
        image_masks = temp_mask[0]*masks
        text_masks = temp_mask[1]*masks
    
    if random_drop_embedding=='image':
        image_masks = masks*(torch.rand(N)>0.5)*1
        text_masks = masks

    return image_masks, text_masks





def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    device = projection_matrix.device  # Get the device of the projection matrix
    x = x.to(device) 
    return x@torch.transpose(projection_matrix, 0, 1)


def inv_project(y, projection_matrix):
    """
    y (Batch*768) should be the CLIP feature (after projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim).  
    this function will return the CLIP penultimate feature. 
    
    Note: to make sure getting the correct penultimate feature, the input y should not be normalized. 
    If it is normalized, then the result will be scaled by CLIP feature norm, which is unknown.   
    """
    return y@torch.transpose(torch.linalg.inv(projection_matrix), 0, 1)




class SynthTextDataset():
    def __init__(self, 
                image_rootdir,
                prob_use_caption=1,
                which_layer_image="after_reproject",
                image_size=512, 
                min_box_size=0.01,
                max_boxes_per_data=8,
                random_drop_embedding ='none',
                max_images=None, # set as 30K used to eval
                random_crop = False,
                random_flip = True,
                ):
        self.image_size = image_size
        self.image_rootdir = image_rootdir
        self.which_layer_image = which_layer_image
        self.prob_use_caption = prob_use_caption
        self.min_box_size = min_box_size
        self.random_drop_embedding = random_drop_embedding
        self.max_boxes_per_data = max_boxes_per_data
        self.max_images = max_images
        self.random_crop = random_crop
        self.random_flip = random_flip
        
        mat_file = os.path.join(image_rootdir, 'gt.mat')
        mat_contents = sio.loadmat(mat_file)
        image_files = extract_image_file_paths(mat_contents, image_rootdir)
        self.image_files = image_files
        self.mat_contents = mat_contents
        self.projection_matrix = torch.load('projection_matrix')
        # preprocessed CLIP feature embedding length: 768  
        self.embedding_len = 768

    def transform_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        if self.random_crop:
            assert False
            arr = random_crop_arr(pil_image, self.image_size) 
        else:
            arr, info = center_crop_arr(image, self.image_size)
		
        info["performed_flip"] = False
        if self.random_flip and random.random()<0.5:
            arr = arr[:, ::-1]
            info["performed_flip"] = True
		
        arr = arr.astype(np.float32) / 127.5 - 1
        arr = np.transpose(arr, [2,0,1])

        return torch.tensor(arr), info 

    def total_images(self):
        return len(self.image_files)

    def get_item_from_tsv(self, index):
        _, item = self.tsv_file[index]
        item = decode_item(item)
        return item

    def mapping(self, image_embedding):
        if self.which_layer_image == 'after':
            # use CLIP image feaure, the aligned feature space with norm=1. 
            return image_embedding
        elif self.which_layer_image == 'after_renorm':
            # same as before but normalize it to 28.7, which is empirically same as text penultimate feature norm.
            return image_embedding*28.7
        elif self.which_layer_image == 'after_reproject':
            # Re-project the CLIP image feature into text penultimate space using text linear matrix and norm it into 28.7
            image_embedding = project( image_embedding.unsqueeze(0), self.projection_matrix.T )
            image_embedding = image_embedding.squeeze(0)
            image_embedding = image_embedding / image_embedding.norm() 
            image_embedding = image_embedding * 28.7 
            return image_embedding


    def __getitem__(self, index):
        if self.max_boxes_per_data > 99:
            assert False, "Are you sure setting such large number of boxes per image?"

        # raw_item = self.get_item_from_tsv(index)
        # is_det = raw_item.get('is_det', False) # if it is from detection (such as o365), then we will make a pseudo caption

        out = {}

        # -------------------- id and image ------------------- # 
        out['id'] = index
        image = self.image_files[index]
        image_tensor, trans_info = self.transform_image(image)
        out["image"] = image_tensor


        # -------------------- grounding token ------------------- # 

        bbox_actual = get_bb(self.mat_contents, index)
        words = get_words(self.mat_contents, index)        
        embeddings = get_embeddings(words)
        captions = get_text_description(self.mat_contents, index)
        areas = []
        all_boxes = []
        all_masks = []
        all_text_embeddings = []
        all_image_embeddings = []
        # if is_det:
        #     all_category_names = []

        # text_embedding_name = 'text_embedding_before' if self.which_layer_text == 'before' else 'text_embedding_after'
        # image_embedding_name = 'image_embedding_after'

        # for anno in annos:
        #     x, y, w, h = anno['bbox']
        #     valid, (x0, y0, x1, y1) = recalculate_box_and_verify_if_valid(x, y, w, h, trans_info, self.image_size, self.min_box_size)

        #     if valid:
        #         areas.append(  (x1-x0)*(y1-y0)  )
        #         all_boxes.append( torch.tensor([x0,y0,x1,y1]) / self.image_size ) # scale to 0-1
        #         all_masks.append(1)
        #         all_text_embeddings.append(anno[text_embedding_name])
        #         all_image_embeddings.append(  self.mapping(anno[image_embedding_name])  )
        #         if is_det:
        #             all_category_names.append(anno["category_name"])
        
        for i, bbox in enumerate(bbox_actual):
            x, y , w, h = bbox
            valid, (x0, y0, x1, y1) = recalculate_box_and_verify_if_valid(x, y, w, h, trans_info, self.image_size, self.min_box_size)
            if valid:
                areas.append(  (x1-x0)*(y1-y0)  )
                all_boxes.append( torch.tensor([x0,y0,x1,y1]) / self.image_size ) # scale to 0-1
                all_masks.append(1)
                all_text_embeddings.append(embeddings[i])
                result_image = mask_image_outside_bbox(image, bbox)
                all_image_embeddings.append(self.mapping(get_clip_image_embeddings(result_image)) )

        # Sort according to area and choose the largest N objects   
        wanted_idxs = torch.tensor(areas).sort(descending=True)[1]
        wanted_idxs = wanted_idxs[0:self.max_boxes_per_data]

        boxes = torch.zeros(self.max_boxes_per_data, 4)
        masks = torch.zeros(self.max_boxes_per_data)
        words = [words[i] for i in wanted_idxs]
        text_embeddings =  torch.zeros(self.max_boxes_per_data, self.embedding_len)
        image_embeddings = torch.zeros(self.max_boxes_per_data, self.embedding_len)
        # if is_det:
        #     category_names = []
        for i, idx in enumerate(wanted_idxs):
            boxes[i] = all_boxes[idx]
            masks[i] = all_masks[idx]
            text_embeddings[i] =  all_text_embeddings[idx]
            image_embeddings[i] = all_image_embeddings[idx]
            # if is_det:
            #     category_names.append(all_category_names[idx])

        if self.random_drop_embedding != 'none':
            image_masks, text_masks = mask_for_random_drop_text_or_image_feature(masks, self.random_drop_embedding)
        else:
            image_masks = masks
            text_masks = masks


        out["boxes"] = boxes
        out["masks"] = masks # indicating how many valid objects for this image-text data
        out["image_masks"] = image_masks # indicating how many objects still there after random dropping applied
        out["text_masks"] = text_masks # indicating how many objects still there after random dropping applied
        out["text_embeddings"] =  text_embeddings  
        out["image_embeddings"] = image_embeddings      
        


        # -------------------- caption ------------------- # 
        if random.uniform(0, 1) < self.prob_use_caption:
            out["caption"] = captions
            # if is_det:
            #     out["caption"] = make_a_sentence(category_names)
            # else:
            #     out["caption"] = raw_item["caption"]
        else:
            out["caption"] = ""

        return out
    
    def vis_getitem_data(self, out, return_tensor=False, print_caption=False):
        """
        Visualize image and boxes from dataset.
        """
        image = out['image']
        boxes = out['boxes']
        # print(boxes)

        # convert image to PIL
        transform = ToPILImage()
        img = transform(image)

        draw = ImageDraw.Draw(img)
        for box in boxes:
            # print("Current Box:", box)
            if sum(box) == 0:
                # print("Skipping box, it's zero.")
                continue
            # Convert float to int if necessary
            box = tuple(map(int, box))
            draw.rectangle(box, outline="red")

        if print_caption:
            print("Image with boxes drawn.")

        return image



    def __len__(self):
        return len(self.image_files)

