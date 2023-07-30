import React, { useRef, useState, useEffect } from 'react';
import {Select,FormControl, MenuItem,InputLabel,FormControlLabel,Dialog, Button,Radio, RadioGroup, Box, Typography,TextField } from '@mui/material';
import { DialogTitle, DialogContent, DialogActions } from '@mui/material';
import 'react-responsive-carousel/lib/styles/carousel.min.css';
import "../style/mapStyle.css";
import DataTable from './DataTable';
import Grid from '@mui/material/Grid';
import SaveSuccessPopup from './SaveSuccessPopup';
import CarouselComponent from './CarouselComponent';
import styled from 'styled-components';
require('dotenv').config()

const MapModule = () => {
  const mapRef = useRef(null);
  const [selectedRectangle, setSelectedRectangle] = useState(null);
  const [rectangles, setRectangles] = useState([]);
  const [drawingManager, setDrawingManager] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingPredict, setIsLoadingPredict] = useState(false);
  const [exportDirectory, setExportDirectory] = useState('');
  const [imagePrefix, setImagePrefix] = useState('');
  const [downloadedImage, setDownloadedImage] = useState(null);
  const [predictedFile, setPredictedFile] = useState(null);
  const [predictedMaksdFile, setPredictedMaksdFile] = useState(null);
  const [models, setModels] = useState([]);
  const [showResults, setShowResults] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [sateliteDowloadType, setSateliteDowloadType] = React.useState('landset_download');
  const [imageName, setImageName] = React.useState('landset_download');
  const [open, setOpen] = useState(false);
  const [showPopup, setShowPopup] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [modelName, setModelName] = useState("landsat");
  const togglePopup = () => {
    setShowPopup(!showPopup);
  };

  useEffect(() => {
    fetchExistingModels();
  }, []);

  const handleChangeSelectedRectangle = (imageNameT,rectangleCoordinates,image,predicted,masked,exportDirectoryPath,imagePrefixPath) => {
    handleRemoveAllClick()
    const rectangle = new window.google.maps.Rectangle({
      bounds: rectangleCoordinates,
      // Other properties...
    });
    const bounds = rectangle.getBounds();
    const center = bounds.getCenter();

    if (mapRef.current) {
      const map = drawingManager.getMap();
      map.panTo(center);
    }
    setSelectedRectangle(rectangle);
    setPredictedFile(predicted);
    setPredictedMaksdFile(masked);
    setExportDirectory(exportDirectoryPath);
    setImagePrefix(imagePrefixPath);
    setImageName(imageNameT);
    setDownloadedImage(image);
    setShowResults(true); // Reset showResults state

    // setShowModal(true);
    setRectangles((prevRectangles) => [...prevRectangles, rectangle]);
    rectangle.setMap(drawingManager.getMap());
    drawingManager.setOptions({
      drawingControl : false,
  });
  };

  const fetchExistingModels = () => {
    fetch('http://127.0.0.1:5000/load_models')
      .then((response) => response.json())
      .then((data) => {
        if (data) {
          setModels(data["models"]);
        } else {
          setModels([]);
        }
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  const handleImageNameChange = (event) => {
    setImageName(event.target.value);
  }

  const handleModelNameChange = (event) => {
    setModelName(event.target.value);
  }

  const handleSateliteChange = (event) => {
    setSateliteDowloadType(event.target.value);
  }

  useEffect(() => {
    const script = document.createElement('script');
    console.log(process.env.REACT_APP_MAPS_API_KEY);
    script.src = `https://maps.googleapis.com/maps/api/js?key=${process.env.REACT_APP_MAPS_API_KEY}&libraries=drawing`;
    script.async = true;
    script.defer = true;
    document.head.appendChild(script);

    script.addEventListener('load', handleMapLoad);

    return () => {
      script.removeEventListener('load', handleMapLoad);
    };
  }, []);

  

  const handleMapLoad = () => {
    const map = new window.google.maps.Map(mapRef.current, {
      center: { lat: 37.7749, lng: -122.4194 },
      disableDefaultUI: true,
      zoom: 6,
      zoomControl: true,
    });

    const manager = new window.google.maps.drawing.DrawingManager({
      drawingMode: null,
      drawingControlOptions: {
        drawingModes: ['rectangle'],
      },
      rectangleOptions: {
        strokeWeight: 0,
        fillOpacity: 0.45,
      },
    });

    window.google.maps.event.addListener(manager, 'overlaycomplete', handleOverlayComplete);

    manager.setMap(map);
    setDrawingManager(manager);
  };

  const handleOverlayComplete = (event) => {
    const rectangle = event.overlay;
    
    window.google.maps.event.addListener(rectangle, 'click', () => {
      removeRectangle(rectangle);
    });
    setRectangles((prevRectangles) => [...prevRectangles, rectangle]);
    setSelectedRectangle(rectangle);
    
  //   drawingManager.setOptions({
  //     drawingControl : false,
  // });

  };

  const removeRectangle = (rectangle) => {
    rectangle.setMap(null);
    setRectangles((prevRectangles) => prevRectangles.filter((r) => r !== rectangle));
    setSelectedRectangle(null);
  };



  const handleDownloadImage = () => {
    if (selectedRectangle) {
      const bounds = selectedRectangle.getBounds().toJSON();
      const coordinates = {
        north: bounds.north,
        south: bounds.south,
        east: bounds.east,
        west: bounds.west,
      };

      setIsLoading(true);

      fetch(`http://127.0.0.1:5000/${sateliteDowloadType}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(coordinates),
      })
        .then((response) => response.json())
        .then((data) => {
          const { image, export_directory, image_prefix } = data;

          setExportDirectory(export_directory);
          setImagePrefix(image_prefix);
          setDownloadedImage(image);
          setShowResults(false); // Reset showResults state
          setShowModal(true);
        })
        .catch((error) => {
          console.error('Error:', error);
        })
        .finally(() => {
          setIsLoading(false);
        });
    }
  };
  const handleClose = () => {
    setOpen(false);
  };

  const handleSaveRes = () => {
    setOpen(true);
  }

  const handleSave = () => {
    if (exportDirectory && imagePrefix && imageName) {
      setIsLoading(true);
      fetch('http://127.0.0.1:5000/save_results', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_name: imageName,
          export_directory: exportDirectory,
          image_prefix: imagePrefix,
          predicted_image: predictedFile,
          predicted_masked_image: predictedMaksdFile,
          img_source: sateliteDowloadType,
          coordinates: selectedRectangle.getBounds().toJSON(),
          downloaded_image:downloadedImage,
          algo_name: modelName
        }),
      })
      .then((response) => response.json())
      .then((data) => {
        setSaveSuccess(!saveSuccess); 
        togglePopup();
      })
        .catch((error) => {
          console.error('Error:', error);
        })
        .finally(() => {
          setIsLoading(false);
          handleClose();
        });
    }
  };

  const handlePredict = () => {
    if (exportDirectory && imagePrefix) {
      setIsLoadingPredict(true);
      fetch('http://127.0.0.1:5000/landset_predict_get', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          export_directory: exportDirectory,
          image_prefix: imagePrefix,
          img_source: sateliteDowloadType,
          algo_name:modelName
        }),
      })
      .then((response) => response.json())
      .then((data) => {
      // Simulating predict completion
        setIsLoading(false);
        const { resized_image, masked_image } = data;
        setPredictedFile(resized_image);
        setPredictedMaksdFile(masked_image);
        setShowResults(true);
      })
        .catch((error) => {
          console.error('Error:', error);
        })
        .finally(() => {
          setIsLoading(false);
          setIsLoadingPredict(false);
        });
    }
  };

  const handleRemoveAllClick = () => {
    setPredictedFile(null);
    setPredictedMaksdFile(null);
    setDownloadedImage(null);
    rectangles.forEach((rectangle) => {
      rectangle.setMap(null);
    });
    setRectangles([]);
    setSelectedRectangle(null);
    drawingManager.setOptions({
      drawingControl : true,
  });
  };

  const handleShowResults = () => {
    setShowModal(true);
  };

  useEffect(() => {
    if (drawingManager) {
      drawingManager.setDrawingMode(null);
    }
  }, [drawingManager]);
    const StyledButton = styled.button`
    margin: 8px 1px;
    margin-top: 40px;
    background-color: #47a99b;
    padding: 12px 24px; /* Adjust the padding values as needed */
    font-size: 16px; /* Adjust the font-size as needed */
    color: white;
    border-radius: 4px;
    &:hover {
      background-color: #7bb6ad;
  `;
  return (
    <div style={{ position: 'relative' }}>
        <Typography variant="h2" style={{  
          display: 'flex',
          marginLeft: '440px',
          padding:'5px',
          fontSize: '24px',
          fontWeight: 'bold',
        }}>
         Prediction
        </Typography>
    <Grid container spacing={0}>
        <Grid item xs={12}>
          <Grid container spacing={2}>
            <Grid item xs={8}>
              <div
                ref={mapRef}
                style={{
                  marginLeft:'20px',
                  height: '450px',
                  background: 'rgba(0,0,0,0.5)',
                }}
              ></div>
            </Grid>
            <Grid item xs={4}>
              <DataTable reload={saveSuccess} changeSelectedRectangle={handleChangeSelectedRectangle}/>
            </Grid>
          </Grid>
        </Grid>
        <Grid container spacing={2} className="btnsGrid" >
            <StyledButton style={{marginLeft:'15px'}} onClick={handleRemoveAllClick} variant="contained">
              Reset
            </StyledButton>
            <FormControl variant="standard"  className="map-select" sx={{ m: 1, minWidth: 130 }}>
            <InputLabel className="map-select-title" id="demo-simple-select-standard-label">Satellite Name</InputLabel>
            <Select
              labelId="demo-simple-select-label"
              id="demo-simple-select-standard"
              value={sateliteDowloadType}
              className="map-select"
              onChange={handleSateliteChange}
              label="Satellite"
            >
              <MenuItem style={{padding:'2px',marginLeft:'7px' ,display:'block'}} value="landset_download">Landsat</MenuItem>
              <MenuItem style={{padding:'2px',marginLeft:'7px' ,display:'block'}} value="geos_download">Geos</MenuItem>
            </Select>
            </FormControl>
            <FormControl variant="standard" className="map-select" sx={{ m: 1, minWidth: 130 }}>
            <InputLabel className="map-select-title" id="demo-simple-select-standard-label" >Model Name</InputLabel>
            <Select 
              labelId="demo-simple-select-standard-label"
              id="demo-simple-select-standard"
              value={modelName}
              className="map-select"
              onChange={handleModelNameChange}
              label="Model"
            >
              {models.map((model) => (
                  <MenuItem style={{padding:'2px',marginLeft:'7px' ,display:'block'}} value={model.modelType}>{model.modelName}</MenuItem>
              ))}
              <MenuItem style={{padding:'2px',marginLeft:'7px' ,display:'block'}} value="edge">Egde Detecion</MenuItem>
            </Select>
            </FormControl>
            <StyledButton onClick={handleDownloadImage} disabled={!selectedRectangle||isLoadingPredict} style={{ display: selectedRectangle ? "block" : "none" }} variant="contained">
              {isLoading ? 'Loading...' : 'Download Image'}
            </StyledButton>
            <StyledButton 
              onClick={handlePredict}
              disabled={!selectedRectangle || isLoadingPredict ||isLoading || !downloadedImage}
              style={{ display: selectedRectangle && downloadedImage ? "block" : "none" }}
              variant="contained"
            >
              {isLoadingPredict ? 'Loading...' : 'Predict'}
            </StyledButton>
          
            {showResults && (<>
              <StyledButton
                onClick={handleShowResults} 
                disabled={isLoading} 
                style={{ display: predictedFile != null ? "block" : "none" }}
                variant="contained">
                Show Results
              </StyledButton>
              <StyledButton
              style={{ display: predictedFile != null ? "block" : "none" }}
               onClick={handleSaveRes} disabled={isLoading} variant="contained">
                Save Results
              </StyledButton>
              <Dialog open={open} onClose={handleClose}>
              <DialogTitle>Enter Image Name</DialogTitle>
              <DialogContent>
                <TextField
                  autoFocus
                  label="Image Name"
                  variant="filled"
                  value={imageName}
                  onChange={handleImageNameChange}
                />
              </DialogContent>
              <DialogActions>
                <Button onClick={handleClose}>Cancel</Button>
                <Button onClick={handleSave} disabled={!imageName}>Save</Button>
              </DialogActions>
            </Dialog>
              <SaveSuccessPopup open={showPopup} handleClose={togglePopup} />
              </>
            )}
        </Grid>
      </Grid>


      <Dialog open={showModal} onClose={() => setShowModal(false)}>
      <CarouselComponent
          downloadedImage={downloadedImage}
          predictedFile={predictedFile}
          predictedMaksdFile={predictedMaksdFile}
        />
      </Dialog>
    </div>
  );

};

export default MapModule;
