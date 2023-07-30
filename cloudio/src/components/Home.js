import React from 'react';
import { Typography, Button, makeStyles } from '@material-ui/core';
import CloudIcon from '@material-ui/icons/Cloud';

const useStyles = makeStyles((theme) => ({
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    marginTop: theme.spacing(8),
  },
  title: {
    marginBottom: theme.spacing(4),
  },
  cloudIcon: {
    fontSize: 100,
    marginBottom: theme.spacing(2),
  },
  image: {
    width: '100%',
    maxWidth: 400,
    marginBottom: theme.spacing(2),
  },
  button: {
    marginTop: theme.spacing(2),
  },
}));

const Home = () => {
  const classes = useStyles();

  return (
    <div className={classes.container}>
      <Typography variant="h4" className={classes.title}>
        Welcome to CLOUD.IO
      </Typography>
      <img src="/clouds.jpeg" alt="Clouds" className={classes.image} style={{ width: '300px', height: '300px' }} />
      <Typography variant="h5" className={classes.title}>
        How to use?
      </Typography>
      <Typography variant="body1" align="center">
        Choose an rectangle on the map ,satellite and model
      </Typography>
      <Typography variant="body1" align="center">
        Cloud detection is a powerful tool that helps identify and classify different types of clouds in images. It can
        be used in various applications, <br/>such as weather forecasting, satellite imagery analysis, and climate research.
      </Typography>
      <Button variant="contained" color="primary" className={classes.button}>
        Get Started
      </Button>
    </div>
  );
};

export default Home;
