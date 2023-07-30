import React, { Component } from "react";
import Login from "./components/Login";
import Particles from "react-particles-js";
import { loadStarsPreset } from "tsparticles-preset-stars";
import { Container } from "tsparticles-react";

class App extends Component {
  // This method customizes the tsParticles installation
  async customInit(engine) {
    // This adds the preset to tsParticles, you can safely use it
    await loadStarsPreset(engine);
  }

  render() {
    const particlesOptions = {
      preset: "stars",
    };

    return (
      <Container>
        <Particles
          id="particles"
          options={particlesOptions}
          init={this.customInit}
        />
        <Login />
      </Container>
    );
  }
}

export default App;
