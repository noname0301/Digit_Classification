import React, { useState, useRef, useEffect } from 'react';

function TestingPage() {
  const [results, setResults] = useState<Array<{ image: string; prediction: number; actual?: number }>>([]);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [apiPort, setApiPort] = useState<number | null>(null);

  useEffect(() => {
    const findApiPort = async () => {
      for (let port = 8000; port < 9000; port++) {
        try {
          const response = await fetch(`http://localhost:${port}/health`);
          if (response.ok) {
            setApiPort(port);
            console.log(`API server found at port ${port}`);
            break;
          }
        } catch (error) {
          console.log("No connection at", port);
          continue;
        }
      }
    };
    findApiPort();
  }, []);


  const handleFileUpload =  async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) return;

    setIsLoading(true);
    const newResults: Array<{ image: string; prediction: number }> = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];

      const formData = new FormData();
      formData.append('file', file, 'digit.png');

      try {
        const response = await fetch(`http://localhost:${apiPort}/predict`, {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          console.log('Upload successful!');
          const data = await response.json();
          const prediction = data.prediction;
          newResults.push({
            image: URL.createObjectURL(file),
            prediction: prediction
          });
        } else {
          console.error('Upload failed');
        }
      } catch (error) {
        console.error('Error processing image:', error);
      }
    }

    setResults([...results, ...newResults]);
    setIsLoading(false);
  };

  const clearResults = () => {
    setResults([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="flex flex-col items-center space-y-6">
      <h1 className="text-3xl font-bold text-primary-600">Test Model</h1>
      <div className="bg-white p-6 rounded-lg shadow-lg w-full max-w-2xl">
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Upload Images
          </label>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            onChange={handleFileUpload}
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-primary-50 file:text-primary-700
              hover:file:bg-primary-100"
          />
        </div>

        {isLoading && (
          <div className="text-center py-4">
            <p className="text-gray-600">Processing images...</p>
          </div>
        )}

        {results.length > 0 && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
              {results.map((result, index) => (
                <div key={index} className="border rounded p-2">
                  <img src={result.image} alt={`Test ${index}`} className="w-full h-auto" />
                  <p className="text-center mt-2">
                    Prediction: <span className="font-bold">{result.prediction}</span>
                  </p>
                </div>
              ))}
            </div>
            <button
              onClick={clearResults}
              className="w-full py-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors"
            >
              Clear Results
            </button>
          </>
        )}
      </div>
    </div>
  );
}

export default TestingPage;