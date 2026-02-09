
import { useState, type ChangeEvent } from 'react';
import Viewer from './components/Viewer';
import { Upload, Play, Download, Brain, BoxSelect, Loader2, Layers } from 'lucide-react';

function App() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [segmentationFile, setSegmentationFile] = useState<File | string | null>(null);
  const [prompt, setPrompt] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [segmentationReady, setSegmentationReady] = useState(false);

  const handleFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      // Reset state on new file
      setSegmentationFile(null);
      setSegmentationReady(false);
    }
  };

  const handleSegmentation = async () => {
    if (!prompt || !imageFile) return;
    setIsProcessing(true);
    setSegmentationReady(false);

    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('prompt', prompt);

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Segmentation failed');
      }

      const blob = await response.blob();
      const file = new File([blob], `segmentation_${imageFile.name}`, { type: 'application/gzip' });

      setSegmentationFile(file);
      setSegmentationReady(true);
    } catch (error) {
      console.error('Error running segmentation:', error);
      // You might want to show an error message to the user here
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 overflow-hidden font-sans selection:bg-indigo-500/30">
      {/* Sidebar Controls */}
      <aside className="w-80 flex-shrink-0 border-r border-slate-800 bg-slate-900/50 backdrop-blur-xl flex flex-col p-6 gap-8 z-10">

        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-600 rounded-lg shadow-lg shadow-indigo-900/50">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
              VoxTell
            </h1>
            <p className="text-xs text-slate-500 font-medium tracking-wide">MEDICAL SEGMENTATION</p>
          </div>
        </div>

        {/* Upload Section */}
        <div className="space-y-4">
          <h2 className="text-sm uppercase tracking-wider text-slate-500 font-semibold text-[10px]">Data Input</h2>

          <div className="relative group">
            <input
              type="file"
              accept=".nii,.nii.gz,.gz"
              onChange={handleFileUpload}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
            />
            <div className={`
              border-2 border-dashed rounded-xl p-8 transition-all duration-300
              flex flex-col items-center justify-center gap-3 text-center
              ${imageFile
                ? 'border-indigo-500/50 bg-indigo-500/10'
                : 'border-slate-700 hover:border-slate-500 bg-slate-800/30 hover:bg-slate-800/50'}
            `}>
              <Upload className={`w-8 h-8 ${imageFile ? 'text-indigo-400' : 'text-slate-400'}`} />
              <div className="space-y-1">
                <p className="text-sm font-medium text-slate-300">
                  {imageFile ? imageFile.name : "Upload .nii file"}
                </p>
                <p className="text-xs text-slate-500">
                  {imageFile ? "Ready for analysis" : "Drag & drop or click to browse"}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Prompt Section */}
        <div className="space-y-4 flex-1">
          <h2 className="text-sm uppercase tracking-wider text-slate-500 font-semibold text-[10px]">Text Prompt</h2>

          <div className="relative">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the region to segment (e.g., 'left ventricle', 'tumor in frontal lobe')..."
              className="w-full h-32 bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 resize-none transition-all"
            />
          </div>

          <button
            onClick={handleSegmentation}
            disabled={!imageFile || !prompt || isProcessing}
            className={`
              w-full py-4 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all shadow-lg
              ${(!imageFile || !prompt)
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-900/20 hover:shadow-indigo-900/40 hover:-translate-y-0.5'}
            `}
          >
            {isProcessing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                Run Segmentation
              </>
            )}
          </button>
        </div>

        {/* Results Section */}
        {segmentationReady && (
          <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-xl flex items-center gap-3">
              <div className="p-1.5 bg-emerald-500/20 rounded-lg">
                <BoxSelect className="w-4 h-4 text-emerald-400" />
              </div>
              <div>
                <p className="text-sm font-medium text-emerald-300">Segmentation Complete</p>
                <p className="text-xs text-emerald-500/70">Mask overlay applied</p>
              </div>
            </div>

            <button className="w-full py-3 border border-slate-700 hover:border-slate-600 hover:bg-slate-800 rounded-xl text-sm font-medium text-slate-300 flex items-center justify-center gap-2 transition-all">
              <Download className="w-4 h-4" />
              Download Mask
            </button>
          </div>
        )}

      </aside>

      {/* Main Content / Viewer */}
      <main className="flex-1 flex flex-col relative bg-gradient-to-br from-slate-950 to-slate-900 min-h-0">

        {/* Viewer Container */}
        <div className="flex-1 p-6 relative min-h-0 min-w-0">
          {!imageFile ? (
            <div className="w-full h-full border border-slate-800 rounded-2xl bg-slate-900/30 flex flex-col items-center justify-center text-slate-600 gap-4">
              <Layers className="w-16 h-16 opacity-20" />
              <p>Upload a NIfTI file to start visualization</p>
            </div>
          ) : (
            <div className="w-full h-full rounded-2xl overflow-hidden shadow-2xl relative border border-slate-800/50">
              <Viewer image={imageFile} segmentation={segmentationFile} />

              {/* Viewer Overlay Controls (Floating) */}
              <div className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-slate-900/80 backdrop-blur-md border border-slate-700/50 px-4 py-2 rounded-full flex items-center gap-4 text-xs font-medium text-slate-400">
                <span className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-slate-500"></span>
                  Axial
                </span>
                <span className="w-px h-4 bg-slate-700"></span>
                <span className="hover:text-white cursor-pointer transition-colors">Reset View</span>
              </div>
            </div>
          )}
        </div>

      </main>
    </div>
  );
}

export default App;
