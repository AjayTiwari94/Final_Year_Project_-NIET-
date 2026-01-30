import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { medicalAPI } from '../api';

const PDFUploader = ({ onPDFExtracted }) => {
  const [uploadedPDF, setUploadedPDF] = useState(null);
  const [extractedData, setExtractedData] = useState(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setUploadedPDF(file);
      setError(null);
      
      // Auto-extract on upload
      setIsExtracting(true);
      try {
        const result = await medicalAPI.extractPDF(file);
        setExtractedData(result);
        onPDFExtracted(result);
      } catch (err) {
        setError(err.response?.data?.detail || 'PDF extraction failed');
        console.error('PDF extraction error:', err);
      } finally {
        setIsExtracting(false);
      }
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    maxFiles: 1
  });

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Upload Medical Report (PDF)</h2>

      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all ${
          isDragActive
            ? 'border-green-500 bg-green-50'
            : 'border-gray-300 hover:border-green-400 hover:bg-gray-50'
        }`}
      >
        <input {...getInputProps()} />
        <div className="space-y-3">
          <div className="text-5xl">📄</div>
          <div>
            <p className="text-lg font-semibold text-gray-700">
              {uploadedPDF ? uploadedPDF.name : 'Drop PDF report here or click to browse'}
            </p>
            {!uploadedPDF && (
              <p className="text-sm text-gray-500 mt-2">
                Upload your medical report for AI analysis
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Loading State */}
      {isExtracting && (
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center space-x-3">
            <svg className="animate-spin h-5 w-5 text-blue-600" viewBox="0 0 24 24">
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
            <span className="text-blue-700 font-semibold">Extracting text from PDF...</span>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-600 text-sm">{error}</p>
        </div>
      )}

      {/* Extracted Data */}
      {extractedData && !isExtracting && (
        <div className="mt-4 space-y-4">
          <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
            <p className="text-green-700 font-semibold flex items-center">
              <span className="mr-2">✓</span>
              PDF successfully extracted!
            </p>
          </div>

          {extractedData.key_points && extractedData.key_points.length > 0 && (
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-bold text-gray-800 mb-3">Key Points:</h3>
              <div className="space-y-2">
                {extractedData.key_points.map((point, index) => (
                  <p key={index} className="text-sm text-gray-700 leading-relaxed">
                    {point}
                  </p>
                ))}
              </div>
            </div>
          )}

          {!extractedData.is_medical_report && (
            <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-yellow-700 text-sm">
                ⚠️ This PDF may not be a medical report. Please ensure you upload the correct document.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PDFUploader;
