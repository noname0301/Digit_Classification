import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';

const DrawingPage: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiPort, setApiPort] = useState<number | null>(null);
  const navigate = useNavigate();

  // Drawing state
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const pointsRef = useRef<{ x: number; y: number }[]>([]);

  // Find API port on component mount
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

  const initializeCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = 280;
    canvas.height = 280;

    // Set drawing style
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
  }, []);

  useEffect(() => {
    initializeCanvas();
  }, [initializeCanvas]);

  const drawLine = useCallback((ctx: CanvasRenderingContext2D, start: { x: number; y: number }, end: { x: number; y: number }) => {
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
  }, []);

  const startDrawing = useCallback((e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e as React.MouseEvent).clientX ? 
      (e as React.MouseEvent).clientX - rect.left : 
      (e as React.TouchEvent).touches[0].clientX - rect.left;
    const y = (e as React.MouseEvent).clientY ? 
      (e as React.MouseEvent).clientY - rect.top : 
      (e as React.TouchEvent).touches[0].clientY - rect.top;

    setIsDrawing(true);
    lastPointRef.current = { x, y };
    pointsRef.current = [{ x, y }];

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x, y);
      ctx.stroke();
    }
  }, []);

  const draw = useCallback((e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !lastPointRef.current) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e as React.MouseEvent).clientX ? 
      (e as React.MouseEvent).clientX - rect.left : 
      (e as React.TouchEvent).touches[0].clientX - rect.left;
    const y = (e as React.MouseEvent).clientY ? 
      (e as React.MouseEvent).clientY - rect.top : 
      (e as React.TouchEvent).touches[0].clientY - rect.top;

    const ctx = canvas.getContext('2d');
    if (ctx) {
      drawLine(ctx, lastPointRef.current, { x, y });
      lastPointRef.current = { x, y };
      pointsRef.current.push({ x, y });
    }
  }, [isDrawing, drawLine]);

  const stopDrawing = useCallback(() => {
    setIsDrawing(false);
    lastPointRef.current = null;
  }, []);

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    setPrediction(null);
    setConfidence(null);
    setError(null);
  }, []);

  const predict = useCallback(async () => {
    if (!apiPort) {
      setError('API server not found. Please make sure the backend is running.');
      return;
    }
  
    const canvas = canvasRef.current;
    if (!canvas) return;
  
    setIsLoading(true);
    setError(null);
  
    try {
      await new Promise<void>((resolve, reject) => {
        canvas.toBlob((blob) => {
          if (!blob) {
            reject(new Error("Canvas toBlob returned null"));
            return;
          }
  
          const img = new Image();
          img.src = URL.createObjectURL(blob);
          img.onload = () => {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const ctx = tempCanvas.getContext('2d');
            if (!ctx) return reject(new Error("No 2D context"));
  
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
  
            ctx.drawImage(img, 0, 0, 28, 28);
  
            tempCanvas.toBlob((processedBlob) => {
              if (!processedBlob) {
                reject(new Error("Processed blob is null"));
                return;
              }
  
              const formData = new FormData();
              formData.append('file', processedBlob, 'digit.png');
  
              fetch(`http://localhost:${apiPort}/predict`, {
                method: 'POST',
                body: formData,
              })
                .then((response) => {
                  if (!response.ok) throw new Error(`Server error (${response.status})`);
                  return response.json();
                })
                .then((result) => {
                  setPrediction(result.prediction);
                  setConfidence(result.confidence);
                  resolve();
                })
                .catch((err) => {
                  setError(err.message);
                  setPrediction(null);
                  setConfidence(null);
                  reject(err);
                });
            }, 'image/png');
          };
        }, 'image/png');
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      setPrediction(null);
      setConfidence(null);
    } finally {
      setIsLoading(false);
    }
  }, [apiPort]);  

  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Draw a Digit</h1>
          <p className="mt-2 text-gray-600">Draw a single digit (0-9) in the box below</p>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex flex-col items-center space-y-6">
            <div className="relative">
              <canvas
                ref={canvasRef}
                className="border-2 border-gray-300 rounded-lg bg-white"
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={stopDrawing}
              />
            </div>

            <div className="flex space-x-4">
              <button
                onClick={clearCanvas}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors"
              >
                Clear
              </button>
              <button
                onClick={predict}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                {isLoading ? 'Predicting...' : 'Predict'}
              </button>
            </div>

            {error && (
              <div className="text-red-600 mt-4">
                {error}
              </div>
            )}

            {prediction !== null && (
              <div className="mt-4 text-center">
                <h2 className="text-2xl font-bold text-gray-900">
                  Prediction: {prediction}
                </h2>
                {confidence !== null && (
                  <p className="text-gray-600 mt-2">
                    Confidence: {(confidence * 100).toFixed(2)}%
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DrawingPage;