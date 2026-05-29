"""
Extract features from the whole slide image tiles using several different settings.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import login, hf_hub_download
from timm.layers import SwiGLUPacked
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoModel

from feature_extraction_utils import TileDataset, custom_collate_fn
    
    
def transform(
    image: np.ndarray, 
    stain_matrix_source: np.ndarray, 
    stain_matrix_target: np.ndarray, 
    maxC_source: np.ndarray, 
    maxC_target: np.ndarray,
) -> np.ndarray:
    """ 
    Apply color normalization on H&E-stained image.

    Args:
        image:  Image to perform stain normalization on.
        stain_matrix_source:  Stain color matrix of source image.
        stain_matrix_target:  Stain color matrix of target image.
        maxC_source:  Concentrations of source image.
        maxC_target:  Concentrations of target image.

    Returns:
        normalized_image:  Image after stain normalization.
    """
    # convert RGB to OD
    masked_tile = image
    masked_tile[image == 0] = 1
    OD = np.maximum(-1 * np.log(masked_tile / 255), 1e-6).reshape((-1, 3))
    
    # get source concentration
    source_concentrations = np.linalg.lstsq(stain_matrix_source.T, OD.T, rcond=-1)[0].T

    # get residuals
    OD_reconstructed = np.dot(source_concentrations, stain_matrix_source)
    residuals = OD - OD_reconstructed

    # normalize concentrations
    source_concentrations *= maxC_target / maxC_source

    # get target optical densities
    OD_target = np.dot(source_concentrations, stain_matrix_target)
    # add residuals to target optical densities
    
    if False:
        OD_target = OD_target + residuals

    # convert OD back to RGB
    trans = 255 * np.exp(-1 * OD_target)

    # ensure between 0 and 255
    trans[trans > 255] = 255  # noqa: PLR2004
    trans[trans < 0] = 0
    normalized_image = trans.reshape(image.shape).astype(np.uint8)

    return normalized_image


# define extraction settings
center = 'UMCU'
model_name = 'UNI'
index_stain_norm_target = 4
orientation_norm = False

# get the correct tile size
if model_name == 'UNI':
    tile_dimension = 224
elif model_name == 'UNI2':
    tile_dimension = 224
elif model_name == 'Virchow2':
    tile_dimension = 224
elif model_name == 'H-optimus-0':
    tile_dimension = 224
elif model_name == 'H-optimus-1':
    tile_dimension = 224
elif model_name == 'CONCHv1.5':
    tile_dimension = 512
else:
    NotImplementedError

# define paths and settings
if True:
    tile_information_subfolder = f"{center}_tessellation_{tile_dimension}"+("_center" if orientation_norm else "")
    tile_information_folder = Path("")/center/tile_information_subfolder
    specimen_information_path = Path(r"")
    stain_information_path = Path("")/center/f"stain_dictionary_{center}_updated.json"
    image_folder = Path(r"")/center
    addition = f'_stain_norm_{index_stain_norm_target}'
    output_folder = Path(r"")/center/model_name/(tile_information_subfolder+addition)
    model_folder = Path(r"")/model_name


# define target stain references
stain_cluster_centers = {
    'UMCU': [
        {   "name": "",
            "stain_matrix_target": [[0.5157266685841214, 0.7621781470098861, 0.3912933369375211],
                                    [0.24442166037857652, 0.8762503914319566, 0.41526293291494554]],
            "maxC_target": [[0.8196921939525796, 0.977769248286716]],
        },
        {    "name": "",
            "stain_matrix_target": [[0.5001435422985296, 0.8072301918728259, 0.31342599513448116],
                                    [0.15526740355139473, 0.9500206426666341, 0.2708372424569486]],
            "maxC_target": [[0.9911380814008711, 0.781860504423114]],
        },
        {    "name": "",
             "stain_matrix_target": [[0.4425685027636072, 0.8203431426580127, 0.3621743346173975],
                                     [0.15091087348807505, 0.8860877760074016, 0.43826289082390163]],
            "maxC_target": [[1.3282404177628417, 0.7946565902492361]]
        },
        {   "name": "",
            "stain_matrix_target": [[0.570350475570727, 0.762831599558755, 0.304611696641604],
                                    [0.2857940624737606, 0.9160920970836309, 0.28124193057874525]],
            "stain_matrix_target_RGB": [[144, 118, 188], [191, 102, 192]],
            "maxC_target": [[1.3643178441694734, 0.9216917552025838]]
        },
        {   "name": "",
            "stain_matrix_target": [[0.5940163941439949, 0.7169927413622437, 0.3647820339901911],
                                    [0.3627789664022076, 0.8416776429278484, 0.39996270692599234]],
            "maxC_target": [[1.244599431171137, 0.5867127739467656]]
        },
    ],
    'CLB': [
        {   "name": "",
            "stain_matrix_target": [[0.522812246017434, 0.7218012937977899, 0.4535088176497155],
                                    [0.30112585738830866, 0.8205336519117424, 0.48584744940416813]],
            "maxC_target": [[0.5479670498596079, 0.4310740124098098]]
        },
        {   "name": "",
            "stain_matrix_target": [[0.5095984056885539, 0.7206740051245819, 0.4700408953031409],
                                    [0.3142353500667878, 0.6931472914906761, 0.6486932842781271]],
            "maxC_target": [[0.7934285579584331, 0.5415040751162712]]
        },
        {   "name": "",
            "stain_matrix_target": [[0.568735860564673, 0.6887812119243802, 0.44957753837101655],
                                    [0.3953420675083525, 0.5890939341198603, 0.7047503007742578]],
            "maxC_target": [[0.6701218554550724, 0.45361641839681305]]
        },
        {   "name": "",
            "stain_matrix_target": [[0.5928793861584896, 0.682562274393462, 0.4273204594249576],
                                    [0.5115596062354076, 0.6427984392915883, 0.5701902627304091]],
            "maxC_target": [[0.6705076987520094, 0.36604855742721115]]
        },
        {   "name": "",
            "stain_matrix_target": [[0.5179201372007912, 0.7311105889761957, 0.4441126413071282],
                                    [0.27616604319165244, 0.5699968086208351, 0.7738449164722409]],
            "maxC_target": [[0.8447118519529222, 0.7665605735585947]]
        },      
    ],
}


# define other settings
batch_size = 1
magnification = 20
num_workers = 4
pin_memory = False
device = 'cuda'


if __name__ == '__main__':

    # login to Huggingface hub with user access token
    login() 

    # create output folder is it does not exist yet
    if not output_folder.exists():
        os.makedirs(output_folder)
    
    # load specimen information
    df = pd.read_excel(specimen_information_path)

    # load stain information
    with open(stain_information_path, 'r') as f:
        stain_dictionary = json.loads(f.read())

    image_filenames = os.listdir(image_folder)

    normalizatiom_param = None
    if model_name == 'UNI':
        # define normalization parameters
        normalization_param = {'mean': (0.485, 0.456, 0.406), 'stdev': (0.229, 0.224, 0.225)}
        # create model folder is it does not exist yet
        if not output_folder.exists():
            os.makedirs(model_folder)

        # download the model parameters
        hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=model_folder, force_download=True)
        model = timm.create_model(
            "vit_large_patch16_224", 
            img_size=224, 
            patch_size=16, 
            init_values=1e-5, 
            num_classes=0, 
            dynamic_img_size=True,
        )
        model.load_state_dict(torch.load(os.path.join(model_folder, "pytorch_model.bin"), map_location="cpu"), strict=True)
    elif model_name == 'UNI2':
        # define normalization parameters
        normalization_param = {'mean': (0.485, 0.456, 0.406), 'stdev': (0.229, 0.224, 0.225)}
        # create model folder is it does not exist yet
        if not output_folder.exists():
            os.makedirs(model_folder)
        # download the model parameters
        hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=model_folder)
        # define model settings
        timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        model = timm.create_model(
            pretrained=False, **timm_kwargs
        )
        model.load_state_dict(torch.load(os.path.join(model_folder, "pytorch_model.bin"), map_location="cpu"), strict=True)

    elif model_name == 'CONCHv1.5':
        # define normalization parameters
        normalization_param = {'mean': (0.485, 0.456, 0.406), 'stdev': (0.229, 0.224, 0.225)}
        # create model folder is it does not exist yet
        model_path = model_folder/'CONCHv1-5'
        if not model_path.exists():
            os.makedirs(model_path)
        # get the model
        model, _ = AutoModel.from_pretrained(
            'MahmoodLab/TITAN', 
            trust_remote_code=True, 
            cache_dir=model_path,
        ).return_conch()
    elif model_name == 'Virchow2':
        # define normalization parameters
        normalization_param = {'mean': (0.485, 0.456, 0.406), 'stdev': (0.229, 0.224, 0.225)}
        # create model folder is it does not exist yet
        model_path = model_folder/'Virchow2'
        if not model_path.exists():
            os.makedirs(model_path)
        # need to specify MLP layer and activation function for proper init
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2", 
            pretrained=True, 
            mlp_layer=SwiGLUPacked, 
            act_layer=torch.nn.SiLU,
        )
    elif model_name == 'H-optimus-0':
        # define normalization parameters
        normalization_param = {'mean': (0.707223, 0.578729, 0.703617), 'stdev': (0.211883, 0.230117, 0.177517)}
        # create model folder is it does not exist yet
        model_path = model_folder/'H-optimus-0'
        if not model_path.exists():
            os.makedirs(model_path)
        # initialize model
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0", 
            pretrained=True, 
            init_values=1e-5, 
            dynamic_img_size=False,
        )
    elif model_name == 'H-optimus-1':
        # define normalization parameters
        normalization_param = {'mean': (0.707223, 0.578729, 0.703617), 'stdev': (0.211883, 0.230117, 0.177517)}
        # create model folder is it does not exist yet
        model_path = model_folder/'H-optimus-1'
        if not model_path.exists():
            os.makedirs(model_path)
        # initialize model
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1", 
            pretrained=True, 
            init_values=1e-5, 
            dynamic_img_size=False,
        )
    else:
        raise NotImplementedError

    model = model.eval()
    model = model.to(device)

    # configure normalization function
    if normalization_param is not None:
        normalize = Normalize(
            mean=normalization_param['mean'],
            std=normalization_param['stdev'],
        )
    else:
        normalize = None

    # get the correct stain norm target
    if index_stain_norm_target is not None:
        if center == 'UMCU':
            stain_norm_target = stain_cluster_centers['CLB'][index_stain_norm_target]
        else:
            stain_norm_target = stain_cluster_centers['UMCU'][index_stain_norm_target]

    # loop over the tiles and convert to feature vectors
    with torch.no_grad():
        for i, tile_filename in enumerate(os.listdir(tile_information_folder)):
            # skip if feature already exists
            if (output_folder/tile_filename.replace('.json', '.pth')).exists():
                print(f'{i} - {tile_filename.replace('.json', '.pth')} skipped.')
                continue

            # get stain information for image
            try:
                stain_norm_source = stain_dictionary[tile_filename.replace('.json', '.png')]
            except KeyError:
                print(f"Stain information for {tile_filename.replace('.json', '.png')} was not found.")
                continue
            
            # get the name and find the paths to the images            
            name = '_'.join(tile_filename.split('_')[:-1])
            image_paths = [image_folder/p for p in image_filenames if name in p]

            # get the label
            if not len(df[df['slide_name'] == name]):
                continue
            label = df[df['slide_name'] == name]['label'].iloc[0]

            # load tile information
            with open(tile_information_folder/tile_filename) as f:
                tile_dictionary = json.loads(f.read())

            # add information about whether the coordinate represents 
            # the center point of the tile or the top left point
            if 'center' in tile_information_subfolder:
                tile_dictionary['center'] = True
            else:
                tile_dictionary['center'] = False
            # define the dataset and dataloader instances
            tile_dataset = TileDataset(
                image_path=image_paths, 
                tile_information=tile_dictionary,
                magnification=magnification,
                channels_first=True,
            )
            dataloader = DataLoader(
                dataset=tile_dataset,
                batch_size=batch_size,
                sampler=SequentialSampler(tile_dataset),
                collate_fn=custom_collate_fn,
            )

            # loop over the tiles and convert to feature vectors
            feature_dict = {'position': [], 'feature': [], 'label': label}
            for item in tqdm(dataloader):
                # check if item is None
                if item is None:
                    continue
                tiles, positions = item

                if False:
                    plt.imshow(np.transpose(tiles[0, ...].numpy(), (1,2,0)))
                    plt.show()

                # skip the tile if necessary
                if tiles is None:
                    continue
                # apply stain normalization
                if index_stain_norm_target is not None:
                    normalized_tiles = []
                    for i in range(tiles.shape[0]):
                        tile = (np.transpose(tiles[i, ...].numpy(), (1,2,0))*255).astype(np.uint8)
                        if True:
                            plt.imshow(tile)
                            plt.show()
                        tile = transform(
                            tile, 
                            np.array(stain_norm_source['stain_matrix_target']), 
                            np.array(stain_norm_target['stain_matrix_target']), 
                            np.array(stain_norm_source['maxC_target']), 
                            np.array(stain_norm_target['maxC_target']),
                        )
                        if True:
                            plt.imshow(tile)
                            plt.show()
                        normalized_tiles.append(torch.from_numpy(np.transpose(tile, (2,0,1))[None, ...])/255)
                    tiles = torch.concatenate(normalized_tiles, dim=0)

                # apply normalization (that the feature extraction model expects)
                if normalize is not None:
                    tiles = normalize(tiles.to(torch.float))

                # extract the feature
                features = model(tiles.to(device)).to('cpu')
                if model_name == 'Virchow2':
                    class_token = features[:, 0]
                    patch_tokens = features[:, 5:]
                    features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
                
                # save the feature
                feature_dict['position'].append(positions)
                feature_dict['feature'].append(features)
            
            # convert to torch tensors
            feature_dict['feature'] = torch.concatenate(feature_dict['feature'])
            feature_dict['position'] = torch.concatenate(feature_dict['position'])

            # save to file
            torch.save(feature_dict, output_folder/(tile_filename.replace('.json', '.pth')))

            # close slideloader instance
            tile_dataset.loader.close()