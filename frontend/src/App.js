import React from 'react';
import './App.css';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import ImageSection from './components/ImageSection';
import About from './components/About';
import Dataset from './components/Dataset';
import Workflow from './components/workflow';
import ImageUpload from './components/ImageUpload';
import ZZFeatureMap from './components/ZZFeatureMap';





function App() {
  return (
    <div className="App">
      <Navbar />
      <Hero />
      <ImageSection />
      <About />
      <Dataset />
      <Workflow />
      <ZZFeatureMap />
      <ImageUpload />
    </div>
  );
}

export default App;
