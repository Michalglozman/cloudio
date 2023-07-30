import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Button, makeStyles } from '@material-ui/core';
import HomeIcon from '@material-ui/icons/Home';
import AssignmentIcon from '@material-ui/icons/Assignment';
import LockIcon from '@material-ui/icons/Lock';
import CloudIcon from '@mui/icons-material/Cloud';

const useStyles = makeStyles((theme) => ({
  appBar: {
    marginBottom: theme.spacing(3),
    background: 'black', // Set the background color to black
  },
  title: {
    flexGrow: 1,
    marginLeft: theme.spacing(2),
  },
  link: {
    color: theme.palette.common.white,
    textDecoration: 'none',
    marginRight: theme.spacing(2),
  },
}));

const Navigation = () => {
  const classes = useStyles();
  const navigate = useNavigate();

  const handleLogout = () => {
    // Perform logout logic here (clear local storage, redirect, etc.)
    localStorage.clear();
    navigate('/login');
  };

  return (
    <AppBar position="static" className={classes.appBar}>
      <Toolbar>
        <Typography variant="h6" className={classes.title}>
          <img src="/logo-cloudio.png" alt="Clouds" className={`${classes.image} nav-image`} />
        </Typography>
        <Button component={Link} to="/" color="inherit" startIcon={<HomeIcon />} className={classes.link}>
          Home
        </Button>
        <Button component={Link} to="/map" color="inherit" startIcon={<CloudIcon />} className={classes.link}>
          Predict
        </Button>
        <Button component={Link} to="/models" color="inherit" startIcon={<AssignmentIcon />} className={classes.link}>
          Models Management
        </Button>
        {localStorage.getItem('userId') ? (
          <Button onClick={handleLogout} color="inherit" startIcon={<LockIcon />} className={classes.link}>
            Logout
          </Button>
        ) : (
          <Button component={Link} to="/login" color="inherit" startIcon={<LockIcon />} className={classes.link}>
            Login
          </Button>
        )}
      </Toolbar>
    </AppBar>
  );
};

export default Navigation;
