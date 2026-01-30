import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { medicalAPI } from '../api';

const ImageUploader = ({ selectedImageType, onPredictionComplete, setLoading }) => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setUploadedImage(file);
      setPreview(URL.createObjectURL(file));
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.bmp']
    },
    maxFiles: 1
  });

  const handleAnalyze = async () => {
    if (!uploadedImage) {
      setError('Please upload an image first');
      return;
    }

    setIsAnalyzing(true);
    setLoading(true);
    setError(null);

    try {
      const result = await medicalAPI.predictImage(uploadedImage, selectedImageType);
      onPredictionComplete(result);
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed. Please try again.');
      console.error('Prediction error:', err);
    } finally {
      setIsAnalyzing(false);
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Upload Medical Image</h2>

      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all ${
          isDragActive
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
        }`}
      >
        <input {...getInputProps()} />
        {preview ? (
          <div className="space-y-4">
            <img
              src={preview}
              alt="Preview"
              className="max-h-64 mx-auto rounded-lg shadow-md"
            />
            <p className="text-sm text-gray-600">
              Click or drag to replace image
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="text-6xl">📤</div>
            <div>
              <p className="text-lg font-semibold text-gray-700">
                {isDragActive ? 'Drop image here' : 'Drag & drop or click to upload'}
              </p>
              <p className="text-sm text-gray-500 mt-2">
                Supported: PNG, JPG, JPEG, BMP
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-600 text-sm">{error}</p>
        </div>
      )}

      {/* Analyze Button */}
      <button
        onClick={handleAnalyze}
        disabled={!uploadedImage || isAnalyzing}
        className={`w-full mt-6 py-3 px-6 rounded-lg font-semibold text-white transition-all ${
          !uploadedImage || isAnalyzing
            ? 'bg-gray-400 cursor-not-allowed'
            : 'bg-blue-600 hover:bg-blue-700 active:scale-95'
        }`}
      >
        {isAnalyzing ? (
          <span className="flex items-center justify-center">
            <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
                fill="none"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
            Analyzing...
          </span>
        ) : (
          '🔍 Analyze Image'
        )}
      </button>
    </div>
  );
};

export default ImageUploader;
