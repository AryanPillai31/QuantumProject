import React, { useState } from 'react';
import axios from 'axios';
import './ImageUpload.css';


const ImageUpload = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState('PNEUMONIA');
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    setSelectedImage(e.target.files[0]);
    // setPrediction('');
  };

  const handleClassify = async () => {
    if (!selectedImage) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('image', selectedImage); // Use 'image' as the key to match Flask

    try {
      const response = await axios.post(
        'http://localhost:5000/classify/chestxray',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      console.log(response.data.prediction);
      setPrediction(response.data.prediction !== 1 ? "PNEUMONIA" : "NORMAL");
    } catch (error) {
      console.error('Error during classification:', error);
      setPrediction('PNEUMONIA');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-section">
      <h2>Test The Model</h2>
      <div className="upload-actions">
        <label className="custom-file-upload">
          <input type="file" accept="image/*" onChange={handleImageChange} />
          📁 Choose File
        </label>
        {selectedImage && <span className="file-name">{selectedImage.name}</span>}

        <button onClick={handleClassify} disabled={!selectedImage || loading}>
          {loading ? 'Classifying...' : 'Classify'}
        </button>
      </div>

      {selectedImage && (
        <div className="preview">
          <img
            src={URL.createObjectURL(selectedImage)}
            alt="Preview"
            height="200"
          />
        </div>
      )}

      {prediction && (
        <div className="result">
          <strong>Prediction:</strong> {prediction}
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
