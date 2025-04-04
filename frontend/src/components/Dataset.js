import React from 'react';
import './Dataset.css';

const Dataset = () => {
  return (
    <section className="dataset-section">
      
      <div
        className="qm-health-hero"
        style={{ backgroundImage: `url("/health.png")` }}
      ></div>
        <div className="qm-health-overlay">
          <h3>Quantum Machine Learning in Healthcare</h3>
          <p>
            Quantum machine learning has the potential to revolutionize healthcare by enabling
            faster diagnosis, advanced imaging analysis, and highly accurate predictive models.
            Leveraging quantum computing power can accelerate the processing of large-scale
            medical datasets, improve pattern recognition in radiology, and support early
            detection of diseases such as tuberculosis.
          </p>
        
      </div>

      <div className="dataset-card">
        <h2>Dataset Used</h2>
        <div className="dataset-content">
          <div className="dataset-image">
            <img src="/xray-sample.jpeg" alt="Chest X-ray sample" />
          </div>
          <div className="dataset-info">
            <ul>
              <li><strong>Name:</strong> Tuberculosis (TB) Chest X-ray Dataset</li>
              <li><strong>Source:</strong> NIH & Shenzhen No.3 Hospital</li>
              <li><strong>Size:</strong> 800+ images (400 normal, 400 TB positive)</li>
              <li><strong>Format:</strong> PNG/JPG with label annotations</li>
              <li><strong>Use Case:</strong> Training SVM to detect TB-infected lungs</li>
              <li><strong>Preprocessing:</strong> Rescaling, normalization, grayscale conversion</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Dataset;
