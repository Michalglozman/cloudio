import React from 'react';
import ReactDOM from 'react-dom';
import AppRouter from './Routers/AppRouter';
import { StyledEngineProvider } from "@mui/material/styles";
import { BrowserRouter  } from "react-router-dom";
import './App.css';

ReactDOM.render(
    <React.StrictMode>
      <StyledEngineProvider injectFirst>
        <BrowserRouter>
            <AppRouter />
        </BrowserRouter>
      </StyledEngineProvider>
    </React.StrictMode>,
    document.getElementById("root")
  );
  
