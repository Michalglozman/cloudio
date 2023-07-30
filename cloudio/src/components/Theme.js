import React, { useEffect, useRef } from 'react';
import "../canvas.css";

const Starfield = ({ quantity }) => {
  const canvasRef = useRef(null);
  const stars = [];

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Define the Star class
    class Star {
      constructor(x, y, offset, duration = 100, size = 2) {
        this.x = x;
        this.y = y;
        this.duration = duration;
        this.offset = offset;
        this.size = size;
        this.timer = offset % duration;
      }

      draw() {
        if (this.timer > this.duration) {
          this.timer = 0;
        }
        this.timer += 1;

        const framesize = Math.abs((this.timer / this.duration) - 0.5) * this.size + this.size / 10;

        ctx.beginPath();
        ctx.arc(this.x, this.y, framesize, 0, Math.PI * 2, false);
        ctx.fillStyle = 'white';
        ctx.fill();
      }
    }

    // Spawn stars
    for (let i = 0; i < quantity; i++) {
      const positionX = window.innerWidth * Math.random();
      const positionY = window.innerHeight * Math.random();
      const offset = Math.random() * 100;
      const duration = Math.random() * 50 + 50;
      const size = Math.random() * 7;

      stars.push(new Star(positionX, positionY, offset, duration, size));
    }

    // Animate stars
    const renderFrame = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (let i = 0; i < quantity; i++) {
        stars[i].draw();
      }

      setTimeout(renderFrame, 100);
    };

    // Set canvas size
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    // Start animation
    renderFrame();
  }, [quantity]);

  return <canvas class="canvas" ref={canvasRef} style={{ position: 'fixed', top: 0, left: 0, zIndex: -1 }}></canvas>;
};

export default Starfield;
