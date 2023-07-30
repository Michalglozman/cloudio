import React, { useState } from "react";
import Logo from "../style/clouds.jpeg";
import axios from "axios";
import { Grid, Paper, Button, makeStyles, TextField } from "@material-ui/core";
import { Navigate } from "react-router-dom";

const useStyles = makeStyles((theme) => ({
  paperStyle: {
    padding: 20,
    minHeight: "78vh",
    height: "78vh",
    width: 300,
    margin: "20px auto",
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    color: "white",
  },
  buttonStyle: {
    margin: "8px 0",
    marginTop: "40px",
    backgroundColor: "#47a99b",
    "&:hover": {
      backgroundColor: "#7bb6ad",
    },
  },
  logoStyle: {
    marginLeft: "24%",
    width: "150px",
    height: "120px",
  },
  titleStyle: {
    fontWeight: "bold",
    color: "#4fb5a7",
  },
  textFieldStyle: {
    "& label.Mui-focused": {
      color: "#4fb5a7",
    },
    "& .MuiInput-underline:after": {
      borderBottomColor: "#4fb5a7",
    },
    "& input": {
      color: "white",
    },
    "& .MuiFormControl-root": {
      marginBottom: "12px",
    },
    "& .MuiFormControl-root:first-child": {
      marginBottom: "20px",
    },
  },
}));

const Login = () => {
  const classes = useStyles();
  const [userId, setUserId] = useState("");
  const [password, setPassword] = useState("");
  const [redirect, setRedirect] = useState(false);

  const handleChange = (event) => {
    const { name, value } = event.target;
    if (name === "userId") {
      setUserId(value);
    } else if (name === "password") {
      setPassword(value);
    }
  };

  const handleSubmit = () => {
    const url = "http://127.0.0.1:5000/login";
    const userData = {
      userId,
      password,
    };
    axios
      .get(url, {
        params: userData,
        withCredentials: true, // Add this line
      })
      .then((res) => {
        setUserId("");
        setPassword("");
        localStorage.setItem("userId", res.data.user.userId);
        localStorage.setItem("userName", res.data.user.userName);
        localStorage.setItem("userType", res.data.user.userType);
        localStorage.setItem("token", res.data.accessToken);
        setRedirect(true);
      })
      .catch((err) => console.log(err));
  };
  return (
    <Grid>
      {redirect ? <Navigate to="/" /> : null}
      <Paper elevation={10} className={classes.paperStyle}>
        <Grid align="center"></Grid>
        <img src='clouds.jpeg' alt="Logo" className={classes.logoStyle} />
        <h2 className={classes.titleStyle}>Log in</h2>
        <TextField
          label="Username"
          placeholder="Enter Email"
          fullWidth
          required
          focused
          InputProps={{
            style: { backgroundColor: 'transparent' }
          }}
          name="userId"
          value={userId}
          onChange={handleChange}
          className={classes.textFieldStyle}
        />
        <TextField
          label="Password"
          placeholder="Enter password"
          type="password"
          fullWidth
          required
          focused
          name="password"
          value={password}
          onChange={handleChange}
          className={classes.textFieldStyle}
        />
        <Button
          type="submit"
          color="primary"
          variant="contained"
          className={classes.buttonStyle}
          fullWidth
          onClick={handleSubmit}
        >
          Log in
        </Button>
      </Paper>
    </Grid>
  );
};

export default Login;
