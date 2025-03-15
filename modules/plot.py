from global_land_mask import globe
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from shapely.geometry import Point
from geopandas import GeoDataFrame
import geopandas as gpd

from modules.loss import HaversineLoss

haversineLoss = HaversineLoss()

url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"


mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
normalize = transforms.Normalize(mean.tolist(), std.tolist())

unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

def plot_predictions(model, dataloader, num_samples=5):
    model.eval()
    with torch.no_grad():
        for images, coordinates, image_path in dataloader:

            images = images.cuda()

            outputs = model(images, coordinates, image_path)

            images = images.cpu()
            coordinates = coordinates.cpu()
            outputs = outputs

            rand_index = random.sample(range(0, len(images) - 1), min(num_samples, len(images) - 1))

            for i in range(min(num_samples, len(images))):
                i = rand_index[i]

                pred_lat, pred_lon = outputs['pred'][i].cpu().numpy()
                true_lat, true_lon = coordinates[i].numpy()

                haver_err = haversineLoss(
                                      torch.tensor(np.array([[pred_lon, pred_lat]]), dtype=torch.float32).deg2rad(),
                                      torch.tensor(np.array([[true_lon, true_lat]]), dtype=torch.float32).deg2rad(),
                                      )
                # Display the image
                img = images[i]
                img = unnormalize(img).permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)

                plt.imshow(img)
                plt.title(f'Pred: ({pred_lat:.4f}, {pred_lon:.4f})\nTrue: ({true_lat:.4f}, {true_lon:.4f})\n Haversine: {haver_err}')
                plt.axis('off')

                # World map for better understanding of how bad our prediction is
                geometry = [Point(pred_lon, pred_lat), Point(true_lon, true_lat)]
                geo_df = GeoDataFrame(geometry = geometry)
                world = gpd.read_file(url)
                geo_df.plot(ax=world.plot(color="lightgrey", figsize=(10, 6)), marker='x', c=['red', 'green'], markersize=50);

                plt.show()
            break