import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Button,
  FormControl,
  FormControlLabel,
  Radio,
  RadioGroup,
  Modal,
  TextField,
} from '@mui/material';
import styled from 'styled-components';
import '../style/modelsStyle.css';

const StyledButton = styled.button`
background-color: #47a99b;
padding: 12px 24px; /* Adjust the padding values as needed */
font-size: 16px; /* Adjust the font-size as needed */
color: white;
border-radius: 4px;
border: none;
&:hover {
  background-color: #7bb6ad;
`;
function ModelManagement() {
  const [models, setModels] = useState([]);
  const [modelName, setModelName] = useState('');
  const [modelType, setModelType] = useState('LandSat 8');
  const [modelFile, setModelFile] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  useEffect(() => {
    fetchExistingModels();
  }, []);

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

  const handleModelNameChange = (event) => {
    setModelName(event.target.value);
  };

  const handleModelTypeChange = (event) => {
    setModelType(event.target.value);
  };

  const handleModelFileChange = (event) => {
    setModelFile(event.target.files[0]);
  };

  const handleUpload = () => {
    if (!modelFile) {
      console.error('No model file selected.');
      return;
    }

    const formData = new FormData();
    formData.append('modelFile', modelFile);
    formData.append('modelName', modelName);
    formData.append('modelType', modelType);

    fetch('http://127.0.0.1:5000/upload_models', {
      method: 'POST',
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        setModelName('');
        setModelType('LandSat 8');
        setModelFile(null);
        fetchExistingModels();
        closeModal();
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  const openModal = () => {
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  return (
      <div>
    <Box className="model-management-container">
      <Typography variant="h5" component="h2" gutterBottom className="model-management-title">
        Existing Models
      </Typography>
      <Table className="model-management-table">
        <TableHead>
          <TableRow>
            <TableCell>Model Name</TableCell>
            <TableCell>Model Type</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {models.map((model) => (
            <TableRow key={model._id}>
              <TableCell>{model.modelName}</TableCell>
              <TableCell>{model.modelType}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      {models.length === 0 && (
        <Typography variant="body1" color="textSecondary" className="model-management-message">
          No models found.
        </Typography>
      )}

      <StyledButton
        variant="contained"
        onClick={openModal}
        color="primary"
        className="model-management-upload-btn"
      >
        Add Model
      </StyledButton>
    </Box>

    <Modal open={isModalOpen} onClose={closeModal} className="model-management-modal">
        <Box className="model-management-modal-content">
        <div className="modal-management-title-container">
          <Typography variant="h5"  className="modal-management-title">
            Add Model
          </Typography>
          </div>
          <FormControl className="model-management-form">
            <TextField
              label="Model Name"
              value={modelName}
              onChange={handleModelNameChange}
              className="model-management-input"
            />
            <RadioGroup
              aria-label="Model Type"
              name="modelType"
              value={modelType}
              onChange={handleModelTypeChange}
              className="model-management-radio"
            >
              <FormControlLabel value="landsat" control={<Radio />} label="LandSat 8" />
              <FormControlLabel value="geos" control={<Radio />} label="Geos 18" />
            </RadioGroup>
            <input
              type="file"
              accept=".h5,.hdf5,.pth"
              onChange={handleModelFileChange}
              className="model-management-input"
            />
            <StyledButton
              variant="contained"
              onClick={handleUpload}
              color="primary"
              className="model-management-upload-btn"
            >
              Upload Model
            </StyledButton>
          </FormControl>
        </Box>
      </Modal>
    </div>
  );
}

export default ModelManagement;
