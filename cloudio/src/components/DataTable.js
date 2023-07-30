import React, { useState, useEffect } from 'react';
import { DataGrid } from '@mui/x-data-grid';
import IconButton from '@mui/material/IconButton';
import DeleteIcon from '@mui/icons-material/Delete';
import MapIcon from '@mui/icons-material/Map';
import ShowIcon from '@mui/icons-material/Visibility';
import { ThemeProvider, createTheme } from '@mui/material/styles';

const DataTable = ({ reload,changeSelectedRectangle }) => {
  const [imagesData, setImagesData] = useState([]);

  useEffect(() => {
    fetchExistingImages();
  }, [reload]);

  const handleDelete = (row) => {
      console.log(row.id);
    fetch(`http://127.0.0.1:5000/delete_image/${row.id}`, {
      method: 'DELETE',
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          // Filter out the deleted item from imagesData state
          const updatedImagesData = imagesData.filter((image) => image.id !== row.id);
          setImagesData(updatedImagesData);
        } else {
          console.error('Failed to delete the image');
        }
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  const fetchExistingImages = () => {
    fetch('http://127.0.0.1:5000/load_images')
      .then((response) => response.json())
      .then((data) => {
        if (data) {
          setImagesData(data["images"]);
        } else {
          setImagesData([
            { id: 1, imageName: 'Image 1', date: '2023-06-28' },
            { id: 2, imageName: 'Image 2', date: '2023-06-29' },
          ]);
        }
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  const fillRowsFromImageData = () => {
    return imagesData.map((image) => ({
      id: image.id,
      imageName: image.image_name,
      date: image.update_date,
      coordinates: image.coordinates,
      image_prefix:image.image_prefix,
      algo_name:image.algo_name,
      downloaded_image:image.downloaded_image,
      predicted_image:image.predicted_image,
      predicted_masked_image:image.predicted_masked_image,
      export_directory:image.export_directory
    }));
  };


  const handleShowResults = (row) => {
    changeSelectedRectangle(row.imageName,
        row.coordinates,
        row.downloaded_image,
        row.predicted_image,
        row.predicted_masked_image,
        row.export_directory,
        row.image_prefix
        );
    // (rectangleCoordinates,image,predicted,masked,exportDirectoryPath,imagePrefixPath)
  };


  const columns = [
    { field: 'imageName', headerName: 'Image Name', width: 150 },
    { field: 'date', headerName: 'Date', width: 150 },
    {
      field: 'showOnMap',
      headerName: 'Show on Map',
      width: 100,
      renderCell: (params) => (
        <>
          <IconButton onClick={() => handleShowResults(params.row)}>
            <ShowIcon style={{ color: '#ffffff' }}/>
          </IconButton>
          <IconButton onClick={() => handleDelete(params.row)}>
            <DeleteIcon style={{ color: '#ffffff' }}/>
          </IconButton>
        </>
      ),
    },
  ];

  const rows = fillRowsFromImageData();

  const darkTheme = createTheme({
    palette: {
      mode: 'dark',
    },
  });

  return (
    <ThemeProvider theme={darkTheme}>
      <div style={{ height: '100%', width: '100%', backgroundColor: 'rgba(0, 0, 0, 0.8)' }}>
        <DataGrid rows={rows} columns={columns} pageSize={7} initialState={{
          pagination: {
            paginationModel: {
              pageSize: 6,
            },
          },
        }}
        pageSizeOptions={[6]}
        disableRowSelectionOnClick />
      </div>
    </ThemeProvider>
  );
};

export default DataTable;
