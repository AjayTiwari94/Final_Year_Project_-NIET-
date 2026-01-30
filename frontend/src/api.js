import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API methods
export const medicalAPI = {
  // Get available models
  getModels: async () => {
    const response = await api.get('/api/models');
    return response.data;
  },

  // Predict from image
  predictImage: async (file, imageType) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('image_type', imageType);

    const response = await api.post('/api/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Extract PDF
  extractPDF: async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/api/pdf/extract', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Chat with AI
  chat: async (message, diagnosisContext = null, pdfContext = null) => {
    const response = await api.post('/api/chat', {
      message,
      diagnosis_context: diagnosisContext,
      pdf_context: pdfContext,
    });
    return response.data;
  },

  // Clear chat history
  clearChat: async () => {
    const response = await api.post('/api/chat/clear');
    return response.data;
  },
};

export default api;
