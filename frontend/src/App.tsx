import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import DrawingPage from './pages/DrawingPage';
import TestingPage from './pages/TestingPage';
import EvaluationPage from './pages/EvaluationPage';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <div className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<DrawingPage />} />
            <Route path="/testing" element={<TestingPage />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;