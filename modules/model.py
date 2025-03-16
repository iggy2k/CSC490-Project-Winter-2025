import torchvision
from torchvision import transforms

import torch
import pandas as pd
from torch import  Tensor
import torch.nn as nn

from config import *
from preprocessing.geo_utils import *
from preprocessing.utils import *
GEOCELL_PATH = 'data/geocells_yfcc.csv'

# CLIP_MODEL = 'openai/clip-vit-base-patch32'

CLIP_MODEL = 'geolocal/StreetCLIP'

from transformers import CLIPVisionModel, CLIPProcessor
embed_model = CLIPVisionModel.from_pretrained(CLIP_MODEL)
embed_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
# import clip

# embed_model, embed_processor = clip.load('ViT-B/16', device)

#TODO: move over more stuff from PIGEON

class GeoLocationModel(nn.Module):
    def __init__(self, refiner):
        super(GeoLocationModel, self).__init__()

        self.panorama = False
        # self.hidden_size = embed_dim
        self.serving = False
        self.should_smooth_labels = False
        self.multi_task = False
        # self.heading = heading
        self.yfcc = None
        self.freeze_base = False
        self.hierarchical = False
        self.num_candidates = 5
        self.refiner = refiner

        self.transform = transforms.Compose([
            transforms.Resize((224, 224))
        ])

        # Save variables
        self.base_model = embed_model.base_model
        self.processor = embed_processor

        # Setup
        self._set_hidden_size()
        geocell_path = GEOCELL_PATH
        self.lla_geocells = self.load_geocells(geocell_path)
        self.num_cells = self.lla_geocells.size(0)

        self.input_dim = self.hidden_size

        self.cell_layer = nn.Linear(self.input_dim, self.num_cells)
        self.softmax = nn.Softmax(dim=-1)

        # Freeze / load parameters
        # self._freeze_params()

        # Loss
        self.loss_fnc = nn.CrossEntropyLoss()



    def forward(self, x: Tensor, labels: Tensor, image_path: str):
        # imgs = [np.array(self.transform(Image.open(path))) for path in image_path]
        # imgs_np = np.stack(imgs)
        imgs = [self.transform(Image.open(path)) for path in image_path]
        imgs_np = imgs

        pixel_values = self.processor(images=imgs_np, return_tensors='pt')['pixel_values']
        pixel_values = pixel_values.cuda()

        embedding = self.base_model(pixel_values=pixel_values)

        if self.mode == 'transformer':
            embedding = embedding.last_hidden_state
            embedding = torch.mean(embedding, dim=1)
        else:
            embedding = embedding.pooler_output
        output = embedding

        # Linear layer
        logits = self.cell_layer(output)
        # print('output', output)
        geocell_probs = self.softmax(logits)

        # Compute coordinate prediction
        geocell_preds = torch.argmax(geocell_probs, dim=-1)
        pred_LLH = torch.index_select(self.lla_geocells.data, 0, geocell_preds)

        # Get top 'num_candidates' geocell candidates
        geocell_topk = torch.topk(geocell_probs, self.num_candidates, dim=-1)

        # Soft labels based on distance
        distances = haversine_matrix(labels, self.lla_geocells.data.t())
        label_probs = smooth_labels(distances)

        loss_clf = self.loss_fnc(logits, label_probs)

        loss_refine, pred_LLH, preds_geocell = self.refiner(embedding,
                                          initial_preds=pred_LLH,
                                          candidate_cells=geocell_topk.indices,
                                          candidate_probs=geocell_topk.values)

        return {'pred': pred_LLH, 'loss': loss_clf, 'loss_refine' : loss_refine, 'preds_geocell': preds_geocell}

    def load_geocells(self, path: str) -> Tensor:
        """Loads geocell centroids and converts them to ECEF format

        Args:
            path (str, optional): path to geocells. Defaults to GEOCELL_PATH.

        Returns:
            Tensor: ECEF geocell centroids
        """
        geo_df = pd.read_csv(path)
        lla_coords = torch.tensor(geo_df[['longitude', 'latitude']].values)
        lla_geocells = nn.parameter.Parameter(data=lla_coords, requires_grad=False)
        return lla_geocells

    def _set_hidden_size(self):
        """
        Determines the hidden size of the model
        """
        # self.hidden_size = 512
        if self.base_model is not None:
            try:
                self.hidden_size = self.base_model.config.hidden_size
                self.mode = 'transformer'

            except AttributeError:
                self.hidden_size = self.base_model.config.hidden_sizes[-1]
                self.mode = 'convnext'

    # def _freeze_params(self):
    #     """Freezes model parameters depending on mode
    #     """
    #     if self.base_model is not None:
    #         if self.freeze_base:
    #             for param in self.base_model.parameters():
    #                 param.requires_grad = False

    #         # Load parameters and freeze relevant parameters
    #         elif 'clip-vit' in self.base_model.config._name_or_path and not self.serving:
    #             head = CLIP_PRETRAINED_HEAD_YFCC if self.yfcc else CLIP_PRETRAINED_HEAD
    #             self.load_state(head)
    #             print(f'Initialized model parameters from model: {head}')
    #             for param in self.base_model.vision_model.encoder.layers[:-1].parameters():
    #                 param.requires_grad = False

model = GeoLocationModel()

# No need with accelerate
# model = model.to(device)