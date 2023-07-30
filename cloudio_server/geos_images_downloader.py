import time
import ee
# import google_auth
ee.Initialize()

# Define band names
BLUE = 'CMI_C01'
RED = 'CMI_C02'
VEGGIE = 'CMI_C03'
GREEN = 'GREEN'
COLLECTION_DQF = 'NOAA/GOES/18/FDCC'

# Define the scale and offset function
def applyScaleAndOffset(image):
    names = image.select('CMI_C..').bandNames()
    scales = names.map(lambda name: image.getNumber(ee.String(name).cat('_scale')))
    offsets = names.map(lambda name: image.getNumber(ee.String(name).cat('_offset')))
    scaled = image.select('CMI_C..').multiply(ee.Image.constant(scales)).add(ee.Image.constant(offsets))
    return image.addBands(scaled, overwrite=True)

# Define the function to add a green radiance band
def addGreenBand(image):
    B_BLUE = f'b(\'{BLUE}\')'
    B_RED = f'b(\'{RED}\')'
    B_VEGGIE = f'b(\'{VEGGIE}\')'
    GREEN_EXPR = f'{GREEN} = 0.45 * {B_RED} + 0.10 * {B_VEGGIE} + 0.45 * {B_BLUE}'
    green = image.expression(GREEN_EXPR).select(GREEN)
    return image.addBands(green)

# Define the collection and filters
COLLECTION = 'NOAA/GOES/18/MCMIPM'
START = ee.Date('2022-11-03T20:01:17')
END = START.advance(200, 'minutes')

import random

class GeosEngineExporter:
    def __init__(self):
        self.polygon = None
        self.collection = None

    def set_coordinates(self, cords):
        # Calculate the center coordinates
        center_lon = (cords['west'] + cords['east']) / 2
        center_lat = (cords['south'] + cords['north']) / 2

        # Calculate the maximum difference between lat and lon coordinates
        lat_diff = abs(cords['north'] - cords['south'])
        lon_diff = abs(cords['east'] - cords['west'])
        max_diff = max(lat_diff, lon_diff)

        # Adjust the coordinates to create a square region
        adjusted_cords = {
            'south': center_lat - max_diff / 2,
            'west': center_lon - max_diff / 2,
            'north': center_lat + max_diff / 2,
            'east': center_lon + max_diff / 2
        }
        self.polygon = ee.Geometry.MultiPoint([
            [adjusted_cords["west"], adjusted_cords["south"]],
            [adjusted_cords["west"], adjusted_cords["north"]],
            [adjusted_cords["east"], adjusted_cords["north"]],
            [adjusted_cords["east"], adjusted_cords["south"]],
            [adjusted_cords["west"], adjusted_cords["south"]]
        ])

    def export_images(self):
        img_num = random.randint(0, 39)
        initGeometry = ee.Geometry.Point(self.polygon.coordinates().get(0))
        folder_name = self.get_export_directory()
        collection = ee.ImageCollection(COLLECTION) \
            .filterDate(START, END) \
            .map(applyScaleAndOffset) \
            .map(addGreenBand)

        # Separate the two domains
        domain1_col = collection.filter(ee.Filter.eq('domain', 1))

        # Select the desired image
        selectedImage = ee.Image(domain1_col.toList(domain1_col.size()).get(img_num))

        # Export the image
        exportImage = selectedImage.select('CMI_C02', 'GREEN', 'CMI_C01', 'CMI_C03')
        filename = f'{selectedImage.get("system:index").getInfo()}_4bands'

        task = ee.batch.Export.image.toDrive(
            image=exportImage,
            description=filename,
            folder=folder_name,
            region=domain1_col.geometry(),
            fileFormat='GeoTIFF',
            scale=2500
        )
        task.start()

        while task.status()['state'] == 'RUNNING' or task.status()['state'] == 'READY':
            print(task.status())
            time.sleep(10)
        time.sleep(10)
        print(task.status())

        collection_fire_clouds = 'NOAA/GOES/18/FDCC'
        collection_dqf = ee.ImageCollection(collection_fire_clouds).select('DQF').filterDate(START, END)
        names = domain1_col.aggregate_array('system:index')
        # Get the desired image at a specific index

        listOfImages_dqf = collection_dqf.toList(domain1_col.size())
        selectedImage = ee.Image(listOfImages_dqf.get(img_num))

        # selectedImage = ee.Image(collection_dqf.toList(collection_dqf.size()).get(0))

        selectedImage_temp = selectedImage.updateMask(selectedImage.gte(ee.Image.constant(2)))
        selectedImage_temp2 = selectedImage_temp.updateMask(selectedImage_temp.lte(ee.Image.constant(2)))

        exportImage = ee.Image(selectedImage_temp2).select('DQF').clip(self.polygon)

        # exportImage = selectedImage_temp2.select('DQF')

        filename_qa = ee.String(names.get(img_num)) \
            .cat('_dqf_mask') \
            .getInfo()

        task = ee.batch.Export.image.toDrive(
            image=exportImage,
            description=filename_qa,
            folder=folder_name,
            region=domain1_col.geometry(),
            fileFormat='GeoTIFF',
            scale=2500
        )
        task.start()
        while task.status()['state'] == 'RUNNING' or task.status()['state'] == 'READY':
            print(task.status())
            time.sleep(10)
        time.sleep(10)
        print(task.status())
        filename = filename.split("_")[0]
        return folder_name, filename


    def get_export_directory(self):
        if self.polygon is None:
            raise ValueError("Coordinates not set. Please call 'set_coordinates' method.")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        directory_name = f"export_{timestamp}"
        return f"geos_test_py_{directory_name}"

if __name__=="__main__":
    exporter = GeosEngineExporter()
    exporter.set_coordinates({
    "south": 39.80908184019318,
    "west": -75.79119709208999,
    "north": 41.902805856402274,
    "east": -72.22064045146499
})
    export_directory,file_name_preffix = exporter.export_images()
    print("Export Directory:", export_directory,"file:",file_name_preffix)
