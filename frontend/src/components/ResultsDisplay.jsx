import React from 'react';

const ResultsDisplay = ({ result }) => {
  if (!result) return null;

  const { predicted_class, confidence, all_probabilities } = result;

  // Sort probabilities by value
  const sortedProbs = Object.entries(all_probabilities)
    .sort(([, a], [, b]) => b - a)
    .filter(([, prob]) => prob > 0.01); // Only show >1%

  const getConfidenceColor = (conf) => {
    if (conf >= 0.8) return 'text-green-600 bg-green-50 border-green-200';
    if (conf >= 0.6) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-red-600 bg-red-50 border-red-200';
  };

  const getConfidenceIcon = (conf) => {
    if (conf >= 0.8) return '✓';
    if (conf >= 0.6) return '⚠';
    return '!';
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 animate-fadeIn">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Analysis Results</h2>

      {/* Main Prediction */}
      <div className={`border-2 rounded-lg p-6 mb-6 ${getConfidenceColor(confidence)}`}>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-xl font-bold">Predicted Diagnosis</h3>
          <span className="text-2xl">{getConfidenceIcon(confidence)}</span>
        </div>
        <p className="text-2xl font-bold mb-2">{predicted_class}</p>
        <div className="flex items-center space-x-2">
          <span className="text-sm font-semibold">Confidence:</span>
          <div className="flex-1 bg-white rounded-full h-3 overflow-hidden">
            <div
              className="h-full bg-current transition-all"
              style={{ width: `${confidence * 100}%` }}
            />
          </div>
          <span className="font-bold text-lg">{(confidence * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* All Probabilities */}
      <div>
        <h3 className="font-bold text-lg text-gray-800 mb-3">Detailed Probabilities</h3>
        <div className="space-y-3">
          {sortedProbs.map(([className, probability]) => (
            <div key={className} className="bg-gray-50 rounded-lg p-4">
              <div className="flex justify-between items-center mb-2">
                <span className="font-semibold text-gray-700">{className}</span>
                <span className="font-bold text-blue-600">
                  {(probability * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all"
                  style={{ width: `${probability * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
        <p className="text-sm text-yellow-800">
          <strong>⚠️ Important:</strong> This is an AI-assisted analysis and should not replace
          professional medical advice. Please consult with a qualified healthcare provider for
          proper diagnosis and treatment.
        </p>
      </div>
    </div>
  );
};

export default ResultsDisplay;
