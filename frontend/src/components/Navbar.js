import React from 'react';
import './Navbar.css';

const Navbar = () => {
  return (
    <header className="navbar">
      <div className="logo">
        <h1>Quantum SVM<br /><span style={{ fontSize: '0.8rem' }}></span></h1>
      </div>
      <nav>
        <a href="#">Home</a>
        <a href="#">Workflow</a>
        <a href="#">Research</a>
        <a href="#">Results</a>
        
      </nav>
    </header>
  );
};

export default Navbar;
