import React from 'react';
import { Link } from 'react-router-dom';

function Navbar() {
  return (
    <nav className="bg-primary-600 text-white shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="text-xl font-bold">Digit Recognition</Link>
          <div className="flex space-x-4">
            <Link to="/" className="hover:text-primary-200 transition-colors">Draw</Link>
            <Link to="/testing" className="hover:text-primary-200 transition-colors">Test</Link>
            <Link to="/evaluation" className="hover:text-primary-200 transition-colors">Evaluate</Link>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;