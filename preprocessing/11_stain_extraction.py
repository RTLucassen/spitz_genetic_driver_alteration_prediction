#    Copyright 2026 Ruben T Lucassen, UMC Utrecht, The Netherlands 
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Select regions in the images to be used for the stain extraction/normalization.
"""

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np
import SimpleITK as sitk
from datetime import datetime
from PIL import Image, ImageTk
from tiatoolbox.tools.stainnorm import MacenkoNormalizer


class ImageGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Macenko Normalizer GUI")

        self.canvas_width = 1000
        self.canvas_height = 800
        self.root.geometry(f"{self.canvas_width}x{self.canvas_height + 60}")
        self.root.resizable(False, False)
        self.root.configure(bg="white")

        self.stain_dict = {}

        self.normalizer = MacenkoNormalizer()
        self.image_files = []
        self.idx = 0
        self.image = None
        self.image_with_bands = None
        self.stain_information = None

        self.scale = 1.0
        self.x_offset = 0
        self.y_offset = 0

        self.canvas = tk.Canvas(
            root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white",
            highlightthickness=0
        )
        self.canvas.pack()

        controls = tk.Frame(root, bg="white")
        controls.pack(fill=tk.X)

        btn_frame = tk.Frame(controls, bg="white")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Load images", command=self.load_images).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Previous", command=self.prev_image).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Skip", command=self.skip).pack(side=tk.LEFT, padx=10)

        self.canvas.bind("<Button-1>", self.onclick)
        self.root.bind("<Left>", self._on_prev_key)
        self.root.bind("<Right>", self._on_next_key)
        self.root.bind("<a>", self._on_prev_key)
        self.root.bind("<d>", self._on_next_key)
        self.root.bind("<s>", self._on_skip_key)

        # handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _on_prev_key(self, event):
        self.prev_image()

    def _on_next_key(self, event):
        self.next_image()

    def _on_skip_key(self, event):
        self.skip_image()

    def load_images(self):
        files = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Image files", "*.tif *.tiff *.png *.jpg *.jpeg")]
        )
        if not files:
            return

        self.image_files = list(files)
        self.idx = 0
        self.load_image()

    def load_image(self):
        path = self.image_files[self.idx]

        self.image = sitk.GetArrayFromImage(
            sitk.ReadImage(path)
        ).astype(np.uint8)

        image_crop = self.image[:int(self.image.shape[0] * 0.25), ...]
        self.normalizer.fit(image_crop)
        self.get_stain_information()

        self.update_image_with_bands()
        self.render()

        self.root.title(f"Macenko stain extraction GUI | {Path(path).name}")

    def get_stain_information(self):
        self.stain_information = {
            "stain_matrix_target": self.normalizer.stain_matrix_target.tolist(),
            "stain_matrix_target_RGB": self.normalizer.stain_matrix_target_RGB.tolist(),
            "maxC_target": self.normalizer.maxC_target.tolist(),
        }

    def update_image_with_bands(self):
        band1 = (
            np.ones((30, self.image.shape[1], 3))
            * np.array(self.normalizer.stain_matrix_target_RGB[0])[None, None, ...]
        ).astype(np.uint8)

        band2 = (
            np.ones((30, self.image.shape[1], 3))
            * np.array(self.normalizer.stain_matrix_target_RGB[1])[None, None, ...]
        ).astype(np.uint8)

        self.image_with_bands = np.concatenate([self.image, band1, band2], axis=0)

    def render(self):
        h, w, _ = self.image_with_bands.shape

        scale_w = self.canvas_width / w
        scale_h = self.canvas_height / h
        self.scale = min(scale_w, scale_h)

        new_w = int(w * self.scale)
        new_h = int(h * self.scale)

        self.x_offset = (self.canvas_width - new_w) // 2
        self.y_offset = (self.canvas_height - new_h) // 2

        pil_img = Image.fromarray(self.image_with_bands)
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)

        self.tk_img = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(
            self.x_offset,
            self.y_offset,
            anchor="nw",
            image=self.tk_img
        )

    def onclick(self, event):
        if self.image is None:
            return

        x = int((event.x - self.x_offset) / self.scale)
        y = int((event.y - self.y_offset) / self.scale)

        if x < 0 or y < 0 or x >= self.image.shape[1] or y >= self.image.shape[0]:
            return

        offset = 25
        h, w, _ = self.image.shape

        y0 = max(0, y - offset)
        y1 = min(h, y + offset)
        x0 = max(0, x - offset)
        x1 = min(w, x + offset)

        crop = self.image[y0:y1, x0:x1, :]
        if crop.size == 0:
            return

        self.normalizer.fit(crop)
        self.get_stain_information()

        self.update_image_with_bands()
        self.render()

    def prev_image(self):
        if self.image_files and self.idx > 0:
            self.idx -= 1
            self.load_image()

    def next_image(self):
        if self.image_files and self.idx < len(self.image_files) - 1:
            self.stain_dict[Path(self.image_files[self.idx]).name] = self.stain_information.copy()
            self.idx += 1
            self.load_image()

    def skip(self):
        if self.image_files and self.idx < len(self.image_files) - 1:
            self.stain_dict[Path(self.image_files[self.idx]).name] = "skip"
            self.idx += 1
            self.load_image()

    def on_close(self):
        if self.image_files:
            self.stain_dict[Path(self.image_files[self.idx]).name] = self.stain_information.copy()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path(f"stain_dictionary_{timestamp}.json")

        with open(save_path, "w") as f:
            json.dump(self.stain_dict, f, indent=2)

        self.root.destroy()


if __name__ == "__main__":
    
    root = tk.Tk()
    app = ImageGUI(root)
    root.mainloop()