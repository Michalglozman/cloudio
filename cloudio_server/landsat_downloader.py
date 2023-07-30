import time
import ee

# ee.Authenticate()
ee.Initialize()

class EarthEngineExporter:
    def __init__(self):
        self.cords = None
        self.points = None
        self.collection = None

    def authenticate(self):
        ee.Authenticate()

    def initialize(self):
        ee.Initialize()

    def set_coordinates(self, cords):
        self.cords = cords
        self.points = ee.Geometry.MultiPoint(
            [[cords["west"], cords["south"]],
             [cords["west"], cords["north"]],
             [cords["east"], cords["north"]],
             [cords["east"], cords["south"]],
             [cords["west"], cords["south"]]])

    def get_qa_bits(self, image, start, end, newName):
        pattern = 0
        for i in range(start, end + 1):
            pattern += 2 ** i
        return image.select([0], [newName]).bitwiseAnd(pattern).rightShift(start)

    def export_images(self):
        if self.points is None:
            raise ValueError("Coordinates not set. Please call 'set_coordinates' method.")

        image_collection_name = 'LANDSAT/LC08/C01/T1_TOA'

        collection = ee.ImageCollection(image_collection_name) \
            .filterDate('2020-01-01', '2021-01-01') \
            .select('B4', 'B3', 'B2', 'B5', 'BQA')

        export_directory = self.get_export_directory()

        # Loop over points
        for ii in range(1):
            initGeometry = ee.Geometry.Point(self.points.coordinates().get(ii))
            images = collection.filterBounds(initGeometry)
            first = ee.Image(images.first())

            names = images.aggregate_array('LANDSAT_PRODUCT_ID')
            print('Names of collection images:', names.getInfo())

            count = images.size()
            QA = first.select('BQA')

            clouds = self.get_qa_bits(QA, 4, 4, 'cloud_state').expression("b(0) == 1")
#            print(clouds.getInfo())

            cloud_band = ee.Image(clouds.updateMask(clouds).toFloat())
            # print(cloud_band.getInfo())

            collection_features = ee.FeatureCollection(images)
            filename = ee.String(names.get(0)).cat(ee.String('_properties')).getInfo()
            # print('Feature Collection:', collection_features.getInfo())

            # task_table = ee.batch.Export.table.toDrive(
            #     collection=collection_features,
            #     description=filename,
            #     folder=export_directory,
            #     fileFormat='CSV'
            # )
            # task_table.start()

            task_list = []

            im1 = 0
            imlast = 1

            for jj in range(im1, imlast):
                selectedImage = ee.Image(images.toList(images.size()).get(jj))

                exportImage = selectedImage.select('B4', 'B3', 'B2', 'B5')
                filename = ee.String(names.get(jj)).cat(ee.String('_4bands')).getInfo()
                task_image = ee.batch.Export.image.toDrive(
                    image=exportImage,
                    description=filename,
                    folder=export_directory,
                    region=exportImage.geometry(),
                    fileFormat='GeoTIFF',
                    scale=100
                )
                time.sleep(5)
                task_image.start()
                task_list.append(task_image)

                bands = ['B4', 'B3', 'B2', 'B5']
                for band in bands:
                    exportImage = selectedImage.select(band)
                    filename = ee.String(names.get(jj)).cat(ee.String('_' + band)).getInfo()
                    task_band = ee.batch.Export.image.toDrive(
                        image=exportImage,
                        description=filename,
                        folder=export_directory,
                        region=first.geometry(),
                        fileFormat='GeoTIFF',
                        scale=100
                    )
                    time.sleep(5)
                    task_band.start()
                    task_list.append(task_band)

                exportImage_qa = selectedImage.select('BQA')
                filename = ee.String(names.get(jj)).cat(ee.String('_qa')).getInfo()
                task_qa = ee.batch.Export.image.toDrive(
                    image=exportImage_qa,
                    description=filename,
                    folder=export_directory,
                    region=first.geometry(),
                    fileFormat='GeoTIFF',
                    scale=100
                )
                time.sleep(5)
                task_qa.start()
                task_list.append(task_qa)

                clouds = self.get_qa_bits(exportImage_qa, 4, 4, 'cloud_state').expression("b(0) == 1")
                cloud_band = ee.Image(clouds.updateMask(clouds))
                filename = ee.String(names.get(jj)).cat(ee.String('_clouds')).getInfo()
                task_cloud = ee.batch.Export.image.toDrive(
                    image=cloud_band,
                    description=filename,
                    folder=export_directory,
                    region=first.geometry(),
                    fileFormat='GeoTIFF',
                    scale=100
                )
                time.sleep(5)
                task_cloud.start()
                task_list.append(task_cloud)

                # print(cloud_band.getInfo())

            while any(task.status()['state'] == 'RUNNING' for task in task_list):
                print('Waiting for tasks to complete...')
                time.sleep(10)

            print('All tasks completed.')

            return export_directory, names.get(0)

    def get_export_directory(self):
        if self.cords is None:
            raise ValueError("Coordinates not set. Please call 'set_coordinates' method.")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        directory_name = f"export_{timestamp}"
        return f"earthengine_test_py_{directory_name}"

if __name__=="__main__":
    exporter = EarthEngineExporter()
    exporter.set_coordinates({"south": 45.307709725927566, "west": -73.82865542565793,
                              "north": 45.85364916048188, "east": -72.97721499597043})
    export_directory,file_name_preffix = exporter.export_images()
    print("Export Directory:", export_directory,"file:",file_name_preffix)
