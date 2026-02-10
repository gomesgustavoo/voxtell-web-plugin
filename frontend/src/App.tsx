
import { useState, useRef, type ChangeEvent } from 'react';
import Viewer, { type ViewerRef, SLICE_TYPE } from './components/Viewer';
import { Upload, Play, Download, Brain, Loader2, Layers, Eye, EyeOff, Trash2, Pencil, Save, X } from 'lucide-react';

interface Segmentation {
  id: string;
  file: File;
  prompt: string;
  isVisible: boolean;
  color: string;      // NiiVue colormap name
  displayColor: string; // CSS color for UI
  type: 'prompt' | 'drawn'; // Source of segmentation
}

const SEGMENTATION_COLORS = [
  { nv: 'red', css: '#ef4444' },
  { nv: 'green', css: '#22c55e' },
  { nv: 'blue', css: '#3b82f6' },
  { nv: 'warm', css: '#f59e0b' },       // amber/orange
  { nv: 'cool', css: '#06b6d4' },       // cyan
  { nv: 'violet', css: '#8b5cf6' },     // purple
  { nv: 'winter', css: '#0ea5e9' },     // sky blue
];

function App() {
  const viewerRef = useRef<ViewerRef>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [segmentations, setSegmentations] = useState<Segmentation[]>([]);
  const [prompt, setPrompt] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDrawingMode, setIsDrawingMode] = useState(false);
  const [sliceType, setSliceType] = useState<SLICE_TYPE>(SLICE_TYPE.MULTIPLANAR);

  const handleFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      // Reset state on new file
      setSegmentations([]);
    }
  };

  const handleSegmentation = async () => {
    if (!prompt || !imageFile) return;
    setIsProcessing(true);

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

      const colorObj = SEGMENTATION_COLORS[segmentations.length % SEGMENTATION_COLORS.length];
      const newSeg: Segmentation = {
        id: Date.now().toString(),
        file,
        prompt,
        isVisible: true,
        color: colorObj.nv,
        displayColor: colorObj.css,
        type: 'prompt'
      };

      setSegmentations(prev => [...prev, newSeg]);
      setPrompt(''); // Clear prompt after success
    } catch (error) {
      console.error('Error running segmentation:', error);
      // You might want to show an error message to the user here
    } finally {
      setIsProcessing(false);
    }
  };

  const toggleVisibility = (id: string) => {
    setSegmentations(prev => prev.map(seg =>
      seg.id === id ? { ...seg, isVisible: !seg.isVisible } : seg
    ));
  };

  const deleteSegmentation = (id: string) => {
    setSegmentations(prev => prev.filter(seg => seg.id !== id));
  };

  // Drawing mode handlers
  const toggleDrawingMode = () => {
    if (isDrawingMode) {
      // Exiting drawing mode - discard any unsaved drawing
      viewerRef.current?.clearDrawing();
    }
    setIsDrawingMode(!isDrawingMode);
  };

  const handleSaveDrawing = (file: File) => {
    const colorObj = SEGMENTATION_COLORS[segmentations.length % SEGMENTATION_COLORS.length];
    const newSeg: Segmentation = {
      id: Date.now().toString(),
      file,
      prompt: 'Manual Drawing',
      isVisible: true,
      color: colorObj.nv,
      displayColor: colorObj.css,
      type: 'drawn'
    };
    setSegmentations(prev => [...prev, newSeg]);
    setIsDrawingMode(false);
  };

  const handleSaveCurrentDrawing = async () => {
    await viewerRef.current?.saveDrawing();
  };

  const handleDownload = async () => {
    if (segmentations.length === 0) return;

    // If only one segmentation, download it directly
    if (segmentations.length === 1) {
      const seg = segmentations[0];
      const url = URL.createObjectURL(seg.file);
      const a = document.createElement('a');
      a.href = url;
      a.download = `voxtell_${seg.prompt.replace(/\s+/g, '_')}.nii.gz`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      return;
    }

    // For multiple segmentations, we'll need to use a zip library
    // Import JSZip dynamically or download individually
    // For now, let's download them individually with a small delay
    for (let i = 0; i < segmentations.length; i++) {
      const seg = segmentations[i];
      setTimeout(() => {
        const url = URL.createObjectURL(seg.file);
        const a = document.createElement('a');
        a.href = url;
        a.download = `voxtell_${i + 1}_${seg.prompt.replace(/\s+/g, '_')}.nii.gz`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }, i * 100); // Small delay between downloads to avoid browser blocking
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 overflow-hidden font-sans selection:bg-indigo-500/30">
      {/* Sidebar Controls */}
      <aside className="w-96 flex-shrink-0 border-r border-slate-800 bg-slate-900/50 backdrop-blur-xl flex flex-col p-5 gap-5 z-10 overflow-y-auto">

        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-600 rounded-lg shadow-lg shadow-indigo-900/50">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
              VoxTell WebViewer
            </h1>
            <p className="text-xs text-slate-500 font-medium tracking-wide">AI MEDICAL SEGMENTATION</p>
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
              border-2 border-dashed rounded-xl p-4 transition-all duration-300
              flex flex-col items-center justify-center gap-2 text-center
              ${imageFile
                ? 'border-indigo-500/50 bg-indigo-500/10'
                : 'border-slate-700 hover:border-slate-500 bg-slate-800/30 hover:bg-slate-800/50'}
            `}>
              <Upload className={`w-6 h-6 ${imageFile ? 'text-indigo-400' : 'text-slate-400'}`} />
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
        <div className="space-y-3">
          <h2 className="text-sm uppercase tracking-wider text-slate-500 font-semibold text-[10px]">Text Prompt</h2>

          <div className="relative">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the region to segment (e.g., 'left ventricle', 'tumor in frontal lobe')..."
              className="w-full h-20 bg-slate-800/50 border border-slate-700 rounded-xl p-3 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 resize-none transition-all"
            />
          </div>

          <button
            onClick={handleSegmentation}
            disabled={!imageFile || !prompt || isProcessing}
            className={`
              w-full py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all shadow-lg
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

          {/* Drawing Mode Controls */}
          <div className="pt-4 border-t border-slate-700">
            {!isDrawingMode ? (
              <button
                onClick={toggleDrawingMode}
                disabled={!imageFile}
                className={`
                  w-full py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all
                  ${!imageFile
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                    : 'bg-violet-600 hover:bg-violet-500 text-white shadow-lg shadow-violet-900/20'}
                `}
              >
                <Pencil className="w-4 h-4" />
                Draw Segmentation
              </button>
            ) : (
              <div className="space-y-2">
                <button
                  onClick={handleSaveCurrentDrawing}
                  className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all shadow-lg"
                >
                  <Save className="w-4 h-4" />
                  Save Drawing
                </button>
                <button
                  onClick={toggleDrawingMode}
                  className="w-full py-3 border border-slate-600 hover:bg-slate-800 text-slate-300 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all"
                >
                  <X className="w-4 h-4" />
                  Cancel
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Results Section */}

        {/* Segmentations List */}
        {segmentations.length > 0 && (
          <div className="space-y-4 flex-1 overflow-y-auto">
            <h2 className="text-sm uppercase tracking-wider text-slate-500 font-semibold text-[10px]">Segmentations</h2>
            <div className="space-y-2">
              {segmentations.map((seg) => (
                <div key={seg.id} className="p-3 bg-slate-800/50 border border-slate-700 rounded-xl flex items-center justify-between group hover:border-slate-600 transition-all">
                  <div className="flex items-center gap-3 overflow-hidden">
                    <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: seg.displayColor }} />
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-slate-300 truncate" title={seg.prompt}>
                        {seg.prompt}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => toggleVisibility(seg.id)}
                      className="p-1.5 hover:bg-slate-700 rounded-lg text-slate-500 hover:text-slate-300 transition-colors"
                      title={seg.isVisible ? "Hide" : "Show"}
                    >
                      {seg.isVisible ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                    </button>
                    <button
                      onClick={() => deleteSegmentation(seg.id)}
                      className="p-1.5 hover:bg-red-500/10 rounded-lg text-slate-500 hover:text-red-400 transition-colors"
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Results Section - Download */}
        {segmentations.length > 0 && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            <button
              onClick={handleDownload}
              className="w-full py-3 border border-slate-700 hover:border-slate-600 hover:bg-slate-800 rounded-xl text-sm font-medium text-slate-300 flex items-center justify-center gap-2 transition-all"
            >
              <Download className="w-4 h-4" />
              Download {segmentations.length === 1 ? 'Segmentation' : `All (${segmentations.length})`}
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
              <Viewer
                ref={viewerRef}
                image={imageFile}
                segmentations={segmentations}
                isDrawingMode={isDrawingMode}
                onSaveDrawing={handleSaveDrawing}
                sliceType={sliceType}
                onSliceTypeChange={setSliceType}
              />
            </div>
          )}
        </div>

      </main>
    </div>
  );
}

export default App;
