import React from 'react';
import './About.css';

const About = () => {
  return (
    <section className="about-container">
      <div className="about-card">
        <h2>About Quantum SVM</h2>
        <p>
        Quantum SVM leverages quantum kernels to explore higher dimensional feature spaces more
          efficiently than classical SVM. By applying PCA first, we isolate the most critical features
          of chest X-ray images and reduce noise. Then, the quantum kernel in QSVM can separate classes
          more effectively, potentially boosting accuracy.
        </p>
       
      </div>
    </section>
  );
};

export default About;
