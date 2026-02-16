import { useRef, useEffect, useState, useCallback, useImperativeHandle, forwardRef } from 'react';
import { Niivue, NVImage, SLICE_TYPE } from '@niivue/niivue';
import { ChevronDown, ChevronUp } from 'lucide-react';

interface ViewerProps {
    image?: File | string | null;
    segmentations?: Array<{
        id: string;
        file: File | string;
        color: string;
        isVisible: boolean;
    }>;
    isDrawingMode?: boolean;
    penValue?: number;       // 0 = erase, 1-255 = draw colors
    isFilled?: boolean;      // true = flood fill, false = edge pen
    drawOpacity?: number;    // 0.0 - 1.0
    onSaveDrawing?: (file: File) => void;
    sliceType?: SLICE_TYPE;
    onSliceTypeChange?: (st: SLICE_TYPE) => void;
}

export interface ViewerRef {
    saveDrawing: () => Promise<void>;
    clearDrawing: () => void;
    drawUndo: () => void;
}

const SLICE_OPTIONS: { label: string; value: SLICE_TYPE }[] = [
    { label: 'Axial', value: SLICE_TYPE.AXIAL },
    { label: 'Coronal', value: SLICE_TYPE.CORONAL },
    { label: 'Sagittal', value: SLICE_TYPE.SAGITTAL },
    { label: 'Multi', value: SLICE_TYPE.MULTIPLANAR },
];

// Which fraction axis each slice type controls
const SLICE_AXIS: Record<number, number> = {
    [SLICE_TYPE.AXIAL]: 2,    // Z
    [SLICE_TYPE.CORONAL]: 1,  // Y
    [SLICE_TYPE.SAGITTAL]: 0, // X
};

const Viewer = forwardRef<ViewerRef, ViewerProps>(({
    image,
    segmentations = [],
    isDrawingMode = false,
    penValue = 1,
    isFilled = false,
    drawOpacity = 0.5,
    onSaveDrawing,
    sliceType = SLICE_TYPE.MULTIPLANAR,
    onSliceTypeChange
}, ref) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [nv, setNv] = useState<Niivue | null>(null);
    // Track which segmentations are already loaded by their IDs
    const loadedSegsRef = useRef<Map<string, { volumeIndex: number }>>(new Map());
    // Bottom toolbar visibility
    const [isToolbarVisible, setIsToolbarVisible] = useState(true);
    // Slice position fraction (0-1) for the slider
    const [sliceFrac, setSliceFrac] = useState(0.5);
    // Total number of slices for the current axis
    const [totalSlices, setTotalSlices] = useState(0);
    // Stable ref to updateSliceInfo so image-loading effect doesn't re-trigger on sliceType change
    const updateSliceInfoRef = useRef<() => void>(() => {});

    // Resize handler to eliminate gray bars
    const handleResize = useCallback(() => {
        if (!containerRef.current || !canvasRef.current || !nv) return;

        const { clientWidth, clientHeight } = containerRef.current;
        const dpr = window.devicePixelRatio || 1;

        // Set canvas size to match container exactly
        canvasRef.current.width = clientWidth * dpr;
        canvasRef.current.height = clientHeight * dpr;
        canvasRef.current.style.width = `${clientWidth}px`;
        canvasRef.current.style.height = `${clientHeight}px`;

        nv.resizeListener();
    }, [nv]);

    // Compute total slices for the current axis
    const updateSliceInfo = useCallback(() => {
        if (!nv || nv.volumes.length === 0) return;
        const vol = nv.volumes[0];
        const dims = vol.dims; // [3, nx, ny, nz, ...]
        if (!dims) return;
        const axis = SLICE_AXIS[sliceType as number];
        if (axis !== undefined) {
            setTotalSlices(dims[axis + 1] || 1); // dims is 1-indexed: dims[1]=nx, dims[2]=ny, dims[3]=nz
        }
        // Sync slider with current crosshair position
        const pos = nv.scene.crosshairPos;
        if (pos && axis !== undefined) {
            setSliceFrac(pos[axis]);
        }
    }, [nv, sliceType]);

    // Keep ref in sync with latest callback
    updateSliceInfoRef.current = updateSliceInfo;

    useEffect(() => {
        if (!canvasRef.current || !containerRef.current) return;

        const niivue = new Niivue({
            backColor: [0, 0, 0, 1],
            show3Dcrosshair: true,
        });

        niivue.attachToCanvas(canvasRef.current);
        niivue.setSliceType(niivue.sliceTypeMultiplanar);
        niivue.setMultiplanarLayout(2);
        niivue.setMultiplanarPadPixels(0);

        setNv(niivue);

        return () => {
            // Cleanup
        };
    }, []);

    // Set up ResizeObserver to handle container size changes
    useEffect(() => {
        if (!containerRef.current || !nv) return;

        handleResize();

        const resizeObserver = new ResizeObserver(() => {
            handleResize();
        });

        resizeObserver.observe(containerRef.current);

        return () => {
            resizeObserver.disconnect();
        };
    }, [nv, handleResize]);

    // Effect to sync the slice type from props
    useEffect(() => {
        if (!nv) return;
        nv.setSliceType(sliceType);
        if (sliceType === SLICE_TYPE.MULTIPLANAR) {
            nv.setMultiplanarLayout(2); // GRID
            nv.setMultiplanarPadPixels(0);
        }
        nv.updateGLVolume();
        handleResize();
        updateSliceInfo();
    }, [nv, sliceType, handleResize, updateSliceInfo]);

    // Effect to handle base image
    useEffect(() => {
        if (!nv || !image) return;

        const loadVolume = async () => {
            try {
                nv.volumes = [];
                loadedSegsRef.current.clear();

                if (typeof image === 'string') {
                    await nv.loadVolumes([{ url: image }]);
                } else if (image instanceof File) {
                    const nvImage = await NVImage.loadFromFile({
                        file: image,
                        name: image.name,
                    });
                    nv.addVolume(nvImage);
                }
            } catch (e) {
                console.error("Failed to load image:", e);
            }

            nv.updateGLVolume();
            handleResize();
            updateSliceInfoRef.current();
        };

        loadVolume();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [nv, image, handleResize]);

    // Effect to handle drawing mode toggle
    useEffect(() => {
        if (!nv || !image) return;

        // Wait for volume to be loaded before enabling drawing
        if (nv.volumes.length === 0) return;

        if (isDrawingMode) {
            nv.createEmptyDrawing();
            nv.setDrawingEnabled(true);
            nv.setDrawOpacity(drawOpacity);
            nv.setPenValue(penValue, isFilled);
            console.log('Drawing mode enabled — pen:', penValue, 'filled:', isFilled, 'opacity:', drawOpacity);
        } else {
            nv.setDrawingEnabled(false);
        }
        nv.updateGLVolume();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [nv, isDrawingMode, image]);

    // Sync pen value & fill mode when they change while drawing
    useEffect(() => {
        if (!nv || !isDrawingMode) return;
        nv.setPenValue(penValue, isFilled);
    }, [nv, isDrawingMode, penValue, isFilled]);

    // Sync draw opacity when it changes while drawing
    useEffect(() => {
        if (!nv || !isDrawingMode) return;
        nv.setDrawOpacity(drawOpacity);
        nv.updateGLVolume();
    }, [nv, isDrawingMode, drawOpacity]);

    // Slice navigation handler — called from slider
    const handleSliceChange = useCallback((fraction: number) => {
        if (!nv || nv.volumes.length === 0) return;
        const axis = SLICE_AXIS[sliceType as number];
        if (axis === undefined) return;

        const pos = [...nv.scene.crosshairPos] as [number, number, number];
        pos[axis] = fraction;
        nv.scene.crosshairPos = pos;
        nv.updateGLVolume();
        setSliceFrac(fraction);
    }, [nv, sliceType]);

    // Expose methods through ref
    useImperativeHandle(ref, () => ({
        saveDrawing: async () => {
            if (!nv || !onSaveDrawing) return;

            try {
                // Get the drawing as a NIfTI Uint8Array (no filename = no download)
                const result = await nv.saveImage({
                    filename: '',
                    isSaveDrawing: true,
                    volumeByIndex: 0
                });

                // saveImage returns Uint8Array when successful, boolean on failure
                if (result && result instanceof Uint8Array) {
                    const blob = new Blob([result.buffer as ArrayBuffer], { type: 'application/gzip' });
                    const file = new File(
                        [blob],
                        `manual_segmentation_${Date.now()}.nii.gz`,
                        { type: 'application/gzip' }
                    );
                    onSaveDrawing(file);

                    // Clear the drawing after saving
                    nv.setDrawingEnabled(false);
                    nv.drawBitmap = null;
                }
            } catch (e) {
                console.error("Failed to save drawing:", e);
            }
        },
        clearDrawing: () => {
            if (!nv) return;
            nv.drawBitmap = null;
            nv.createEmptyDrawing();
            nv.refreshDrawing();
        },
        drawUndo: () => {
            if (!nv) return;
            nv.drawUndo();
        }
    }), [nv, onSaveDrawing]);

    // Effect to handle segmentations
    // Uses name-based volume matching instead of fragile numeric indices
    useEffect(() => {
        if (!nv || !image) return;

        // Helper: find a NiiVue volume by the segmentation id (stored as vol.name)
        const findVolumeBySegId = (segId: string) => {
            // Skip index 0 (the base image)
            for (let i = 1; i < nv.volumes.length; i++) {
                if (nv.volumes[i].name === segId) {
                    return nv.volumes[i];
                }
            }
            return null;
        };

        const syncSegmentations = async () => {
            const currentIds = new Set(segmentations.map(s => s.id));
            const loadedIds = new Set(loadedSegsRef.current.keys());

            // 1. Remove segmentations that are no longer in props
            const idsToRemove = [...loadedIds].filter(id => !currentIds.has(id));
            for (const id of idsToRemove) {
                const vol = findVolumeBySegId(id);
                if (vol) {
                    nv.removeVolume(vol);
                }
                loadedSegsRef.current.delete(id);
            }

            // 2. Add new segmentations and update visibility for existing ones
            for (const seg of segmentations) {
                if (loadedSegsRef.current.has(seg.id)) {
                    // Already loaded — just sync visibility
                    const vol = findVolumeBySegId(seg.id);
                    if (vol) {
                        const newOpacity = seg.isVisible ? 0.5 : 0;
                        if (vol.opacity !== newOpacity) {
                            vol.opacity = newOpacity;
                        }
                    }
                } else {
                    // New segmentation — load it
                    try {
                        let vol;
                        const opacity = seg.isVisible ? 0.5 : 0;
                        const colormap = seg.color;

                        if (typeof seg.file === 'string') {
                            vol = await NVImage.loadFromUrl({
                                url: seg.file,
                                colormap: colormap,
                                opacity: opacity,
                                name: seg.id, // tag with seg id for reliable lookup
                            });
                        } else if (seg.file instanceof File) {
                            vol = await NVImage.loadFromFile({
                                file: seg.file,
                                name: seg.id,
                                colormap: colormap,
                                opacity: opacity
                            });
                        }

                        if (vol) {
                            // Ensure the name matches the seg id for lookup
                            vol.name = seg.id;
                            nv.addVolume(vol);
                            loadedSegsRef.current.set(seg.id, { volumeIndex: -1 }); // index not used
                        }
                    } catch (e) {
                        console.error("Failed to load seg:", seg.id, e);
                    }
                }
            }

            nv.updateGLVolume();
        };

        syncSegmentations();
    }, [nv, segmentations, image]);

    // Determine current active view label
    const activeLabel = SLICE_OPTIONS.find(o => o.value === sliceType)?.label ?? 'Multi';
    const isSingleAxis = sliceType !== SLICE_TYPE.MULTIPLANAR && sliceType !== SLICE_TYPE.RENDER;
    const currentSliceNum = Math.round(sliceFrac * Math.max(totalSlices - 1, 1)) + 1;

    return (
        <div
            ref={containerRef}
            className="w-full h-full bg-black rounded-lg overflow-hidden border border-slate-700 shadow-xl"
            style={{ position: 'relative' }}
        >
            <canvas
                ref={canvasRef}
                style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%'
                }}
            />
            {/* Drawing mode indicator */}
            {isDrawingMode && (
                <div className="absolute top-4 left-4 bg-violet-500/90 text-white px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-2 shadow-lg">
                    <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                    Drawing ({activeLabel}) — {penValue === 0 ? 'Eraser' : isFilled ? 'Fill mode' : 'Pen mode'}
                </div>
            )}

            {/* Toolbar toggle button */}
            {image && (
                <button
                    onClick={() => setIsToolbarVisible(prev => !prev)}
                    className="absolute bottom-3 right-3 z-20 p-1.5 bg-slate-800/80 backdrop-blur-sm
                               border border-slate-700/60 rounded-lg text-slate-400
                               hover:text-white hover:bg-slate-700/80 transition-all duration-200 shadow-md"
                    title={isToolbarVisible ? 'Hide controls' : 'Show controls'}
                >
                    {isToolbarVisible ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronUp className="w-3.5 h-3.5" />}
                </button>
            )}

            {/* Bottom control bar */}
            {image && (
                <div className={`
                    absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-3
                    bg-slate-900/85 backdrop-blur-md border border-slate-700/60 rounded-xl px-3 py-2 shadow-lg z-10
                    transition-all duration-300 ease-in-out
                    ${isToolbarVisible
                        ? 'opacity-100 translate-y-0'
                        : 'opacity-0 translate-y-4 pointer-events-none'}
                `}
                    style={{ minWidth: '320px', maxWidth: '90%' }}
                >
                    {/* Slice slider — visible when single-axis view */}
                    {isSingleAxis && totalSlices > 1 && (
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                            <span className="text-[10px] text-slate-400 font-mono whitespace-nowrap w-14 text-center">
                                {currentSliceNum}/{totalSlices}
                            </span>
                            <input
                                type="range"
                                min={0}
                                max={1}
                                step={1 / Math.max(totalSlices - 1, 1)}
                                value={sliceFrac}
                                onChange={(e) => handleSliceChange(parseFloat(e.target.value))}
                                className="flex-1 h-1 accent-indigo-500 cursor-pointer"
                                title={`Slice ${currentSliceNum} of ${totalSlices}`}
                            />
                        </div>
                    )}

                    {/* Divider between slider and axis buttons */}
                    {isSingleAxis && totalSlices > 1 && (
                        <div className="w-px h-5 bg-slate-700/60" />
                    )}

                    {/* Layout / Axis Selector */}
                    <div className="flex items-center gap-1">
                        {SLICE_OPTIONS.map(opt => {
                            const isActive = sliceType === opt.value;
                            return (
                                <button
                                    key={opt.label}
                                    onClick={() => {
                                        onSliceTypeChange?.(opt.value);
                                    }}
                                    className={`
                                        px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all duration-200
                                        ${isActive
                                            ? 'bg-indigo-600 text-white shadow-md shadow-indigo-900/40'
                                            : 'text-slate-400 hover:text-white hover:bg-slate-700/60'}
                                    `}
                                    title={opt.label}
                                >
                                    {opt.label}
                                </button>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
});

Viewer.displayName = 'Viewer';

export { SLICE_TYPE };
export default Viewer;
