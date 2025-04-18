import React, { useState } from "react";
import { Upload } from "lucide-react";
import "./ChestXrayPCA.css";

export default function ChestXrayPCA() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState("chest_xray");
  const handleDatasetChange                   = e => setSelectedDataset(e.target.value);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
    }
  };

  const handleAnalyze = () => {
    if (!preview) return;
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      if (selectedDataset === "chest_xray") {
        runChestXrayAnalysis(image);
      } else {
        runBrainMRIAnalysis(image);
      }
      alert("PCA + QSVM Analysis Complete!");
    }, 2000);
  };

  return (
    <div className="wrapper">
      {/* ===== NAVBAR ===== */}
      <nav className="navbar">
        <div className="navbar-logo">Quantum-SVM </div>
        <ul className="navbar-links">
          <li>
            <a href="#home">Home</a>
          </li>
          <li>
            <a href="#workflow">Workflow</a>
          </li>
          <li>
            <a href="#analysis">Analysis</a>
          </li>
          <li>
            <a href="#info">Info</a>
          </li>
        </ul>
      </nav>

      {/* ===== HOME SECTION ===== */}
      <section id="home" className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">QSVM Analysis</h1>
          <p className="hero-subtitle">
            Harness the power of Principal Component Analysis (PCA) to reduce image dimensionality and
            classify them using a Quantum Support Vector Machine.
          </p>
        </div>
      </section>

      {/* ===== WORKFLOW SECTION ===== */}
      <section id="workflow" className="workflow-section">
        <h2 className="section-title">Project Workflow</h2>
        <div className="workflow-steps">
          <div className="workflow-step">
            <h3>1. Data Preparation</h3>
            <p>
              Collect chest X-ray images for both normal and pneumonia classes. Preprocess them by converting
              to grayscale, resizing, and normalizing.
            </p>
          </div>
          <div className="workflow-step">
            <h3>2. PCA Dimensionality Reduction</h3>
            <p>
              Reduce the dimensionality of the images while preserving the most important features. This helps
              speed up model training and reduces overfitting.
            </p>
          </div>
          <div className="workflow-step">
            <h3>3. QSVM Classification</h3>
            <p>
              Feed the reduced data into a Quantum SVM, leveraging quantum kernels for potentially higher
              feature-space separation and accuracy.
            </p>
          </div>
        </div>
      </section>

      {/* ===== ANALYSIS SECTION ===== */}
      <section id="analysis" className="analysis-section">
        <h2 className="section-title">Upload & Analyze</h2>
        <p className="analysis-subtitle">
          Upload a chest X-ray image to see how PCA + QSVM might process and classify it.
        </p>

        <div className="analysis-container">
          <div className="analysis-card">
            {/* Upload Section */}
            <div className="dataset-selector">
              <label htmlFor="dataset">Choose Dataset:</label>
              <select
                id="dataset"
                value={selectedDataset}
                onChange={handleDatasetChange}
                className="dataset-dropdown"
              >
                <option value="chest_xray">Chest X‑ray (Pneumonia)</option>
                <option value="brain_mri">Brain MRI (Tumor)</option>
              </select>
            </div>

            <label className="upload-box">
              <Upload className="upload-icon" />
              <span className="upload-text">Click or drag to upload a Chest X-ray image</span>
              <input type="file" className="hidden-input" onChange={handleImageUpload} />
            </label>

            {/* Preview of Uploaded Image */}
            {preview && (
              <div className="image-preview">
                <img src={preview} alt="Preview" className="preview-img" />
              </div>
            )}

            {/* Analyze Button */}
            <button className="analyze-button" onClick={handleAnalyze} disabled={loading}>
              {loading ? "Processing..." : "Analyze with QSVM"}
            </button>

            {loading && <p className="loading-text">Performing PCA + QSVM Analysis...</p>}
          </div>
        </div>
      </section>

      {/* ===== INFO SECTION ===== */}
      <section id="info" className="info-section">
        <h2 className="section-title">Why Quantum SVM?</h2>
        <p className="info-text">
          Quantum SVM leverages quantum kernels to explore higher dimensional feature spaces more
          efficiently than classical SVM. By applying PCA first, we isolate the most critical features
          of chest X-ray images and reduce noise. Then, the quantum kernel in QSVM can separate classes
          more effectively, potentially boosting accuracy.
        </p>
        <p className="info-text">
          With sufficient data and hyperparameter tuning, QSVM can match or surpass traditional SVM
          methods in medical imaging tasks.
        </p>
      </section>

      {/* ===== FOOTER ===== */}
      <footer className="footer">
        <p className="footer-text">&copy; 2025 QSVM Medical Imaging | All Rights Reserved.</p>
      </footer>
    </div>
  );
}