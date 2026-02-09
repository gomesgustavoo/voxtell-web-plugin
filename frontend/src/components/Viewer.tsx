import { useRef, useEffect, useState, useCallback, useImperativeHandle, forwardRef } from 'react';
import { Niivue, NVImage } from '@niivue/niivue';

interface ViewerProps {
    image?: File | string | null;
    segmentations?: Array<{
        id: string;
        file: File | string;
        color: string;
        isVisible: boolean;
    }>;
    isDrawingMode?: boolean;
    onSaveDrawing?: (file: File) => void;
}

export interface ViewerRef {
    saveDrawing: () => Promise<void>;
    clearDrawing: () => void;
}

const Viewer = forwardRef<ViewerRef, ViewerProps>(({
    image,
    segmentations = [],
    isDrawingMode = false,
    onSaveDrawing
}, ref) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [nv, setNv] = useState<Niivue | null>(null);
    // Track which segmentations are already loaded by their IDs
    const loadedSegsRef = useRef<Map<string, { volumeIndex: number }>>(new Map());

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

    useEffect(() => {
        if (!canvasRef.current || !containerRef.current) return;

        const niivue = new Niivue({
            backColor: [0, 0, 0, 1],
            show3Dcrosshair: true,
            // Magic Wand (click-to-segment) configuration
            clickToSegment: true,
            clickToSegmentIs2D: true, // 2D only for performance
            clickToSegmentAutoIntensity: true, // Auto-detect intensity threshold
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
        };

        loadVolume();
    }, [nv, image, handleResize]);

    // Effect to handle drawing mode toggle
    useEffect(() => {
        if (!nv || !image) return;

        // Wait for volume to be loaded before enabling drawing
        if (nv.volumes.length === 0) return;

        if (isDrawingMode) {
            // Create empty drawing bitmap when entering draw mode
            nv.createEmptyDrawing();
            nv.setDrawingEnabled(true);
            nv.setPenValue(1); // Pen value for segmentation (1 = red in default colormap)
            console.log('Drawing mode enabled, pen value:', 1);
        } else {
            nv.setDrawingEnabled(false);
        }
    }, [nv, isDrawingMode, image]);

    // Expose methods through ref
    useImperativeHandle(ref, () => ({
        saveDrawing: async () => {
            if (!nv || !onSaveDrawing) return;

            try {
                // Commit any pending grow-cut selection from scroll adjustments
                nv.refreshDrawing(true, true);

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
        }
    }), [nv, onSaveDrawing]);

    // Effect to handle segmentations - OPTIMIZED
    useEffect(() => {
        if (!nv || !image) return;

        const syncSegmentations = async () => {
            const currentIds = new Set(segmentations.map(s => s.id));
            const loadedIds = new Set(loadedSegsRef.current.keys());

            // 1. Remove segmentations that are no longer in props
            const idsToRemove = [...loadedIds].filter(id => !currentIds.has(id));
            if (idsToRemove.length > 0) {
                const volumesToRemove: any[] = [];
                for (const id of idsToRemove) {
                    const info = loadedSegsRef.current.get(id);
                    if (info && nv.volumes[info.volumeIndex]) {
                        volumesToRemove.push(nv.volumes[info.volumeIndex]);
                    }
                    loadedSegsRef.current.delete(id);
                }
                for (const vol of volumesToRemove) {
                    nv.removeVolume(vol);
                }
                rebuildVolumeIndexMap();
            }

            // 2. Add new segmentations and update visibility for existing ones
            for (let i = 0; i < segmentations.length; i++) {
                const seg = segmentations[i];
                const loadedInfo = loadedSegsRef.current.get(seg.id);

                if (loadedInfo) {
                    const vol = nv.volumes[loadedInfo.volumeIndex];
                    if (vol) {
                        const newOpacity = seg.isVisible ? 0.5 : 0;
                        if (vol.opacity !== newOpacity) {
                            vol.opacity = newOpacity;
                        }
                    }
                } else {
                    try {
                        let vol;
                        const opacity = seg.isVisible ? 0.5 : 0;
                        const colormap = seg.color;

                        if (typeof seg.file === 'string') {
                            vol = await NVImage.loadFromUrl({
                                url: seg.file,
                                colormap: colormap,
                                opacity: opacity
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
                            nv.addVolume(vol);
                            loadedSegsRef.current.set(seg.id, {
                                volumeIndex: nv.volumes.length - 1
                            });
                        }
                    } catch (e) {
                        console.error("Failed to load seg:", seg.id, e);
                    }
                }
            }

            nv.updateGLVolume();
        };

        const rebuildVolumeIndexMap = () => {
            loadedSegsRef.current.clear();
            for (let i = 1; i < nv.volumes.length; i++) {
                const vol = nv.volumes[i];
                const matchingSeg = segmentations.find(s => {
                    if (s.file instanceof File) {
                        return vol.name === s.id;
                    }
                    return false;
                });
                if (matchingSeg) {
                    loadedSegsRef.current.set(matchingSeg.id, { volumeIndex: i });
                }
            }
        };

        syncSegmentations();
    }, [nv, segmentations, image]);

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
                <div className="absolute top-4 left-4 bg-indigo-500/90 text-white px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-2 shadow-lg">
                    <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                    Magic Wand - Click to select, scroll to adjust threshold
                </div>
            )}
        </div>
    );
});

Viewer.displayName = 'Viewer';

export default Viewer;

