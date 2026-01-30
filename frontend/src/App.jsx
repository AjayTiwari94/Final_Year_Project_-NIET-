import { useState } from 'react';
import Header from './components/Header';
import ImageUploader from './components/ImageUploader';
import ResultsDisplay from './components/ResultsDisplay';
import PDFUploader from './components/PDFUploader';
import ChatInterface from './components/ChatInterface';
import './App.css';

function App() {
  const [selectedImageType, setSelectedImageType] = useState('aptos');
  const [predictionResult, setPredictionResult] = useState(null);
  const [pdfContext, setPdfContext] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredictionComplete = (result) => {
    setPredictionResult(result);
    setLoading(false);
  };

  const handlePDFExtracted = (extractedData) => {
    setPdfContext(extractedData);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        {/* Image Type Selector */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Select Image Type</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button
              onClick={() => setSelectedImageType('aptos')}
              className={`p-4 rounded-lg border-2 transition-all ${
                selectedImageType === 'aptos'
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-300 hover:border-blue-300'
              }`}
            >
              <div className="text-4xl mb-2">👁️</div>
              <h3 className="font-bold text-lg">Retinal Scan</h3>
              <p className="text-sm text-gray-600">Diabetic Retinopathy Detection</p>
            </button>

            <button
              onClick={() => setSelectedImageType('ham10000')}
              className={`p-4 rounded-lg border-2 transition-all ${
                selectedImageType === 'ham10000'
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-300 hover:border-blue-300'
              }`}
            >
              <div className="text-4xl mb-2">🔬</div>
              <h3 className="font-bold text-lg">Skin Lesion</h3>
              <p className="text-sm text-gray-600">Dermoscopy Analysis</p>
            </button>

            <button
              onClick={() => setSelectedImageType('mura')}
              className={`p-4 rounded-lg border-2 transition-all ${
                selectedImageType === 'mura'
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-300 hover:border-blue-300'
              }`}
            >
              <div className="text-4xl mb-2">🦴</div>
              <h3 className="font-bold text-lg">X-Ray</h3>
              <p className="text-sm text-gray-600">Bone Fracture Detection</p>
            </button>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column - Upload and Results */}
          <div className="space-y-6">
            <ImageUploader
              selectedImageType={selectedImageType}
              onPredictionComplete={handlePredictionComplete}
              setLoading={setLoading}
            />

            {predictionResult && (
              <ResultsDisplay result={predictionResult} />
            )}

            <PDFUploader onPDFExtracted={handlePDFExtracted} />
          </div>

          {/* Right Column - Chat Interface */}
          <div>
            <ChatInterface
              diagnosisContext={predictionResult}
              pdfContext={pdfContext?.sections?.full_text}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
