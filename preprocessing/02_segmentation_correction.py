"""
Annotate low magnification whole slide images in the dataset.
"""

import natsort
from pathlib import Path

from annotation_tool import AnnotationTool


# define paths
images_folder = Path(r"")
annotations_folder = Path(r"")

# define names        
classes = ['Selection', 'Pen marking']

# define settings
rotate_portrait = True
add_layers = True
autosave = True
start_from_back = True


if __name__ == '__main__':

    # get all filenames from annotations if the annotation folder already exists
    if not annotations_folder.exists():
        annotations_folder.mkdir()
        annotation_paths = [] 
    else:
        annotation_paths = list(annotations_folder.iterdir())

    # prepare paths to images (and optionally annotations)
    paths = []
    for image_path in images_folder.iterdir():

        # check if there are annotations available for the image
        search_term = image_path.stem
        hits = []
        for path in annotation_paths:
            if search_term in path.name:
                hits.append(path)
      
        # get the latest annotation version
        hits = natsort.natsorted(hits)

        # add the path to the latest annotation or None
        # if there are no annotations present
        if not len(hits):
            paths.append((image_path, None))
        else:
            paths.append((image_path, hits[-1]))

    if start_from_back:
        paths.reverse()

    # prepare the annotation tool for annotation
    AnnotationTool(
        input_paths=paths,
        layers=classes,
        output_directory=annotations_folder,
        rotate_portrait=rotate_portrait,
        add_layers=add_layers,
        autosave=autosave, 
    )     