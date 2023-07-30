import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navigation from '../components/Navigation';
import Home from '../components/Home';
import ImagePredictions from '../components/ImagePredictions';
import Login from '../components/Login';
import Starfield from '../components/Theme';
import MapModule from '../components/MapComponent';
import ModelManagement from '../components/ModelManagement';

const AppRouter = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(!!localStorage.getItem('userId'));

  useEffect(() => {
    setIsLoggedIn(!!localStorage.getItem('userId'));
  }, []);

  return (
    <div>
      <Navigation />
      <Starfield quantity={200} />
      <Routes>
        <Route path="/" element={<Home />} />

        {isLoggedIn ? (
          <>
            <Route path="/predict" element={<ImagePredictions />} />
            <Route path="/map" element={<MapModule />} />
            <Route path="/models" element={<ModelManagement />} />
          </>
        ) : (
          <Route path="*" element={<Navigate to="/login" />} />
        )}
        <Route path="/login" element={<Login />} />
      </Routes>
    </div>
  );
};

export default AppRouter;
