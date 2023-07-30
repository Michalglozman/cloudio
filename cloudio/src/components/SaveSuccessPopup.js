import React, { useState } from 'react';
import { Snackbar, Alert } from '@mui/material';

const SaveSuccessPopup = ({ open, handleClose }) => {
  return (
    <Snackbar open={open} autoHideDuration={3000} onClose={handleClose}>
      <Alert onClose={handleClose} severity="success" sx={{ width: '100%' }}>
        Results saved successfully!
      </Alert>
    </Snackbar>
  );
};

export default SaveSuccessPopup;
