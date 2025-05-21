import React, { useState, useEffect } from 'react';

interface ConfusionMatrix {
  [key: number]: {
    [key: number]: number;
  };
}

interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
}

function EvaluationPage() {
  const [apiPort, setApiPort] = useState<number | null>(null);
  const [metrics, setMetrics] = useState<Metrics>({
    accuracy: 0,
    precision: 0,
    recall: 0,
    f1Score: 0
  });

  const [confusionMatrix, setConfusionMatrix] = useState<ConfusionMatrix>({});

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

  useEffect(() => {
    const fetchConfusionMatrix = async () => {
      if (apiPort) {
        try {
          const response = await fetch(`http://localhost:${apiPort}/confusion_matrix`);
          if (response.ok) {
            const data = await response.json();
            setConfusionMatrix(data.confusion_matrix);
            setMetrics(data.metrics);
          } else {
            console.error("Failed to fetch confusion matrix");
          }
        } catch (error) {
          console.error("Error fetching confusion matrix:", error);
        }
      }
    }
    fetchConfusionMatrix();
  }, [apiPort]);


  return (
    <div className="flex flex-col items-center space-y-6">
      <h1 className="text-3xl font-bold text-primary-600">Model Evaluation</h1>
      
      <div className="bg-white p-6 rounded-lg shadow-lg w-full max-w-4xl">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <MetricCard title="Accuracy" value={metrics.accuracy} />
          <MetricCard title="Precision" value={metrics.precision} />
          <MetricCard title="Recall" value={metrics.recall} />
          <MetricCard title="F1 Score" value={metrics.f1Score} />
        </div>

        <h2 className="text-xl font-semibold mb-4">Confusion Matrix</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr>
                <th className="border p-2 bg-gray-50">Actual ↓ Predicted →</th>
                {Array.from({ length: 10 }, (_, i) => (
                  <th key={i} className="border p-2 bg-gray-50">{i}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: 10 }, (_, i) => (
                <tr key={i}>
                  <th className="border p-2 bg-gray-50">{i}</th>
                  {Array.from({ length: 10 }, (_, j) => (
                    <td
                      key={j}
                      className={`border p-2 text-center ${
                        i === j ? 'bg-primary-50' : ''
                      }`}
                    >
                      {confusionMatrix[i]?.[j] || 0}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, value }: { title: string; value: number }) {
  return (
    <div className="bg-gray-50 p-4 rounded-lg">
      <h3 className="text-sm font-medium text-gray-500">{title}</h3>
      <p className="mt-1 text-2xl font-semibold text-gray-900">
        {(value * 100).toFixed(1)}%
      </p>
    </div>
  );
}

export default EvaluationPage;