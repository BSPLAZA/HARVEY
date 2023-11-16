import React, { useState, useEffect } from 'react';
import RingLoader from 'react-spinners/RingLoader';
import { Routes, Route, Link, useNavigate } from 'react-router-dom';
import Video from '/Users/akhilreddy/Documents/VSCode/HARVEY/src/frontend/src/pages/Video.js';
import Monitoring from '/Users/akhilreddy/Documents/VSCode/HARVEY/src/frontend/src/pages/Monitoring.js';
import About from '/Users/akhilreddy/Documents/VSCode/HARVEY/src/frontend/src/pages/About.js';
import { Navbar } from './Components/Navbar';

import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import './Welcome.css';

function App() {
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
    }, 2000);
  }, []);

  const handleButtonClick = (path) => {
    navigate(path);
  };

  return (
    <>
      <div className="App">
       
        {loading ? (
          <RingLoader
            color={"#DFCD90"}
            loading={loading}
            size={80}
            aria-label="Loading Spinner"
            data-testid="loader"
          />
        ) : (
          <div className="welcome-container">
            {/* Conditionally render welcome content based on the route */}
            {window.location.pathname === '/' && (
              <>
                <div className="welcome-text" color='white'>
                  <p>Welcome to</p>
                </div>
                <div className="harvey-text" color='white'>
                  <p>HARVEY</p>
                </div>
                <div className="harvey-acronym-text" color='white'>
                  <p>Harvesting Acceleration Robot for Vegetation Enhancement and Yielding</p>
                </div>

                <div className="button-container">
                  <button onClick={() => handleButtonClick('/')}>Home</button>
                  <button onClick={() => handleButtonClick('/video')}>Video</button>
                  <button onClick={() => handleButtonClick('/monitoring')}>Monitoring</button>
                  <button onClick={() => handleButtonClick('/about')}>About</button>
                </div>
              </>
            )}
            {window.location.pathname === '/video' && (
              <>
                <div className="welcome-text" color='white'>
                  <p>Video Page</p>
                </div>

                <div className="button-container">
                  <button onClick={() => handleButtonClick('/')}>Home</button>
                </div>
              </>
            )}
            {window.location.pathname === '/monitoring' && (
              <>
                <div className="welcome-text" color='white'>
                  <p>Monitoring Page</p>
                </div>

                <div className="button-container">
                  <button onClick={() => handleButtonClick('/')}>Home</button>
                </div>
              </>
            )}
            {window.location.pathname === '/about' && (
              <>
                <div className="welcome-text" color='white'>
                  <p>About Page</p>
                </div>

                <div className="button-container">
                  <button onClick={() => handleButtonClick('/')}>Home</button>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Place the Routes component outside the main div */}
      <Routes>
        <Route path="video" element={<Video />} />
        <Route path="monitoring" element={<Monitoring />} />
        <Route path="about" element={<About />} />
      </Routes>
    </>
  );
}

export default App;
