import { useRef, useEffect, useState, useCallback } from 'react';
import { Niivue, NVImage } from '@niivue/niivue';

interface ViewerProps {
    image?: File | string | null;
    segmentations?: Array<{
        id: string;
        file: File | string;
        color: string;
        isVisible: boolean;
    }>;
}

export default function Viewer({ image, segmentations = [] }: ViewerProps) {
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
            backColor: [0, 0, 0, 1], // Pure black to blend with image background
            show3Dcrosshair: true,
        });

        niivue.attachToCanvas(canvasRef.current);
        niivue.setSliceType(niivue.sliceTypeMultiplanar);
        niivue.setMultiplanarLayout(2); // Auto layout for better space utilization
        niivue.setMultiplanarPadPixels(0); // Remove padding between panels

        setNv(niivue);

        return () => {
            // Cleanup
        };
    }, []);

    // Set up ResizeObserver to handle container size changes
    useEffect(() => {
        if (!containerRef.current || !nv) return;

        // Initial resize
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
                // Ensure we start fresh if image changes
                nv.volumes = [];
                loadedSegsRef.current.clear(); // Clear tracked segmentations

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
            handleResize(); // Ensure proper sizing after load
        };

        loadVolume();
    }, [nv, image, handleResize]);

    // Effect to handle segmentations - OPTIMIZED
    useEffect(() => {
        if (!nv || !image) return;

        const syncSegmentations = async () => {
            const currentIds = new Set(segmentations.map(s => s.id));
            const loadedIds = new Set(loadedSegsRef.current.keys());

            // 1. Remove segmentations that are no longer in props
            const idsToRemove = [...loadedIds].filter(id => !currentIds.has(id));
            if (idsToRemove.length > 0) {
                // Need to remove volumes - find them by working backwards
                // Since removal shifts indices, we collect volumes to remove first
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
                // Rebuild index mapping after removal
                rebuildVolumeIndexMap();
            }

            // 2. Add new segmentations and update visibility for existing ones
            for (let i = 0; i < segmentations.length; i++) {
                const seg = segmentations[i];
                const loadedInfo = loadedSegsRef.current.get(seg.id);

                if (loadedInfo) {
                    // Already loaded - just update visibility (opacity)
                    const vol = nv.volumes[loadedInfo.volumeIndex];
                    if (vol) {
                        const newOpacity = seg.isVisible ? 0.5 : 0;
                        if (vol.opacity !== newOpacity) {
                            vol.opacity = newOpacity;
                        }
                    }
                } else {
                    // New segmentation - need to load it
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
                                name: seg.id, // Use ID for tracking
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

        // Helper to rebuild volume index mapping after removals
        const rebuildVolumeIndexMap = () => {
            // Clear and rebuild based on volume names/order
            // We use seg.id as the volume name, so we can match them
            loadedSegsRef.current.clear();
            for (let i = 1; i < nv.volumes.length; i++) {
                const vol = nv.volumes[i];
                // Find matching segmentation by comparing volume properties
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
        </div>
    );
}
