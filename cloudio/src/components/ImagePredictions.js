import React, { useState, useEffect } from 'react';
import {
  Select,
  MenuItem,
  Button,
  CircularProgress,
  Typography,
  Tab,
  Tabs,
  TextField,
  Radio,
  RadioGroup,
  FormControlLabel,
  Box,
} from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles((theme) => ({
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    marginTop: theme.spacing(2),
  },
  formContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    marginBottom: theme.spacing(2),
    textAlign: 'center',
    padding: theme.spacing(2),
    borderRadius: theme.spacing(1),
  },
  inputField: {
    marginBottom: theme.spacing(2),
  },
  uploadButton: {
    marginBottom: theme.spacing(2),
  },
  predictButton: {
    marginBottom: theme.spacing(2),
  },
  imageContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  imageBox: {
    display: 'flex',
    justifyContent: 'center',
    marginBottom: theme.spacing(2),
  },
  image: {
    maxWidth: '100%',
    maxHeight: '400px',
  },
  loadingContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    marginTop: theme.spacing(2),
  },
}));

const ImagePredictions = () => {
  const classes = useStyles();
  const [tabValue, setTabValue] = useState(0);
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedImageName, setSelectedImageName] = useState('');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('');
  const [predictionImage, setPredictionImage] = useState(null);
  const [selectedImagePreview, setSelectedImagePreview] = useState(null);
  const [imageNames, setImageNames] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch('http://127.0.0.1:5000/get_image_names')
      .then((response) => response.json())
      .then((data) => {
        setImageNames(data.image_names);
      })
      .catch((error) => {
        console.log('Error fetching image names:', error);
      });
  }, []);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleImagePathChange = (event) => {
    setSelectedImage(event.target.value);
  };

  const handleImageNameChange = (event) => {
    setSelectedImageName(event.target.value);
  };

  const handleImageUpload = () => {
    if (!selectedImage || !selectedImageName) {
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append('image_file', selectedImage);
    formData.append('image_name', selectedImageName);

    fetch('http://127.0.0.1:5000/upload_image', {
      method: 'POST',
      body: formData,
    })
      .then((response) => response.blob())
      .then((data) => {
        setSelectedImage(null);
        setLoading(false);
        displayImageBlob(data);
      })
      .catch((error) => {
        console.log('Error uploading image:', error);
        setLoading(false);
      });
  };

  const handleImageSelect = (event) => {
    const selectedName = event.target.value;
    setSelectedImageName(selectedName);

    fetch(`http://127.0.0.1:5000/tiff_preview_by_name?tiff_name=${selectedName}`)
      .then((response) => response.blob())
      .then((data) => {
        setSelectedImage(null);
        displayImageBlob(data);
      })
      .catch((error) => {
        console.log('Error fetching image by name:', error);
      });
  };

  const handleAlgorithmSelect = (event) => {
    setSelectedAlgorithm(event.target.value);
  };

  const handlePredictionClick = () => {
    if (!selectedAlgorithm || !selectedImageName) {
      console.log(`here ${selectedAlgorithm} ${selectedImageName}`);
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append('image_file', selectedImageName);
    formData.append('algorithm_name', selectedAlgorithm);

    fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      body: formData,
    })
      .then((response) => response.blob())
      .then((data) => {
        setLoading(false);
        displayPredictedImageBlob(data);
      })
      .catch((error) => {
        console.log('Error fetching prediction:', error);
        setLoading(false);
      });
  };

  const displayImageBlob = (blob) => {
    const imageUrl = URL.createObjectURL(blob);
    setSelectedImagePreview(imageUrl);
  };

  const displayPredictedImageBlob = (blob) => {
    const imageUrl = URL.createObjectURL(blob);
    setPredictionImage(imageUrl);
  };

  return (
    <div className={classes.container}>
      <Tabs value={tabValue} onChange={handleTabChange}>
        <Tab label="Upload Image" />
        <Tab label="Choose Image" />
      </Tabs>

      {tabValue === 0 && (
        <Box className={classes.formContainer}>
          <Typography variant="h6">Upload Image:</Typography>
          <TextField
            value={selectedImage}
            onChange={handleImagePathChange}
            label="Image Path"
            fullWidth
            className={classes.inputField}
          />
          <Typography variant="subtitle1">Image Name:</Typography>
          <TextField
            value={selectedImageName}
            onChange={handleImageNameChange}
            label="Image Name"
            fullWidth
            className={classes.inputField}
          />
          <Button variant="contained" color="primary" onClick={handleImageUpload} className={classes.uploadButton}>
            Upload Image
          </Button>
        </Box>
      )}

      {tabValue === 1 && (
        <Box className={classes.formContainer}>
          <Typography variant="h6">Choose Image:</Typography>
          <Select value={selectedImageName} onChange={handleImageSelect} className={classes.inputField}>
            <MenuItem value="">Select Image</MenuItem>
            {imageNames.map((name) => (
              <MenuItem key={name} value={name}>
                {name}
              </MenuItem>
            ))}
          </Select>
        </Box>
      )}

      <Box className={classes.formContainer}>
        <Typography variant="h6">Choose Algorithm:</Typography>
        <RadioGroup value={selectedAlgorithm} onChange={handleAlgorithmSelect}>
          <FormControlLabel value="lanset" control={<Radio color="primary" />} label="Lanset" />
          <FormControlLabel value="geoset" control={<Radio color="primary" />} label="Geoset" />
        </RadioGroup>
      </Box>

      <Button variant="contained" color="primary" onClick={handlePredictionClick} className={classes.predictButton}>
        Predict
      </Button>

      {loading && (
        <div className={classes.loadingContainer}>
          <CircularProgress />
          <Typography variant="h6">Loading...</Typography>
        </div>
      )}

      <Box className={classes.imageBox}>
        {selectedImagePreview && (
          <div className={classes.imageContainer}>
            <Typography variant="h6">Selected Image:</Typography>
            <img src={selectedImagePreview} alt="Selected image" className={classes.image} />
          </div>
        )}

        {predictionImage && (
          <div className={classes.imageContainer}>
            <Typography variant="h6">Predicted Image:</Typography>
            <img src={predictionImage} alt="Predicted image" className={classes.image} />
          </div>
        )}
      </Box>
    </div>
  );
};

export default ImagePredictions;
