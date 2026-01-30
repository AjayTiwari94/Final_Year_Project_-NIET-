import React from 'react';

const Header = () => {
  return (
    <header className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white shadow-lg">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="text-4xl">🏥</div>
            <div>
              <h1 className="text-3xl font-bold">Medical Imaging Diagnostic System</h1>
              <p className="text-blue-100 text-sm">AI-Powered Medical Image Analysis</p>
            </div>
          </div>
          <div className="hidden md:flex items-center space-x-4">
            <div className="text-right">
              <p className="text-sm text-blue-100">Powered by</p>
              <p className="font-semibold">Deep Learning AI</p>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
