import React from 'react';
import './App.css';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import ImageSection from './components/ImageSection';
import About from './components/About';
import Dataset from './components/Dataset';
import Workflow from './components/workflow';
import ImageUpload from './components/ImageUpload';





function App() {
  return (
    <div className="App">
      <Navbar />
      <Hero />
      <ImageSection />
      <About />
      <Dataset />
      <Workflow />
      <ImageUpload />
    </div>
  );
}

export default App;
