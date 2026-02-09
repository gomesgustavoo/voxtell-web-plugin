import { useRef, useEffect, useState } from 'react';
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
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [nv, setNv] = useState<Niivue | null>(null);

    useEffect(() => {
        if (!canvasRef.current) return;

        const niivue = new Niivue({
            backColor: [0.1, 0.1, 0.1, 1],
            // textHeight: 0.05, // Use default or adjust
        });

        niivue.attachToCanvas(canvasRef.current);
        niivue.setSliceType(niivue.sliceTypeMultiplanar);

        setNv(niivue);

        return () => {
            // Cleanup
        };
    }, []);

    // Effect to handle base image
    useEffect(() => {
        if (!nv || !image) return;

        const loadVolume = async () => {
            // If base image changes, we might want to clear everything
            // But usually we want to keep overlays unless image changes entirely.
            // For now, let's assume image change resets everything (handled in App.tsx by clearing state).

            // Check if volume 0 is already this image to avoid reload? 
            // Simplified: just reload if image prop changes.

            try {
                // Ensure we start fresh if image changes
                nv.volumes = []; // heavy handed but safe

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
        };

        loadVolume();
    }, [nv, image]);

    // Effect to handle segmentations
    useEffect(() => {
        if (!nv || !image) return; // Don't load segs if no base image

        const syncSegmentations = async () => {
            // Strategy: 
            // 1. Identify which segmentations are already loaded in nv.volumes (index > 0).
            // 2. Add new ones.
            // 3. Update visibility for all.
            // 4. Remove ones that are no longer in props.

            // To simplify logic and avoid complex diffing bugs with NiiVue's internal state:
            // We will remove all overlay volumes (index > 0) and reload them. 
            // This is not efficient for large numbers, but robust for this size.
            // Optimization: Only reload if the list LENGTH or content changed. 
            // Visibility changes shouldn't trigger reload.

            // Implementation:
            // Loop through props.segmentations.
            // For each, check if it's already loaded (by some ID/name matching).
            // NiiVue volumes don't preserve our custom IDs easily. 
            // Let's rely on index or re-add.

            // Let's try a smarter update:
            // We look at nv.volumes. We expect volumes[1] to correspond to segmentations[0] etc?
            // No, because we might delete from middle.

            // SAFEST: Remove all overlays and re-add.
            while (nv.volumes.length > 1) {
                nv.removeVolume(nv.volumes[nv.volumes.length - 1]);
            }

            for (const seg of segmentations) {
                try {
                    let vol;
                    // Different colormaps for differentiation?
                    // NiiVue supports 'red', 'green', 'blue', 'warm', 'cool', etc.
                    // We passed a color string. We might need to map it or create a custom lookup table if we want exact hex.
                    // For now, let's assume valid NiiVue colormap names or use standard ones.

                    const opacity = seg.isVisible ? 0.5 : 0;
                    const colormap = seg.color;

                    if (typeof seg.file === 'string') {
                        vol = await NVImage.loadFromUrl({ url: seg.file, colormap: colormap, opacity: opacity });
                    } else if (seg.file instanceof File) {
                        // @ts-ignore
                        vol = await NVImage.loadFromFile({
                            file: seg.file,
                            name: seg.file.name, // or seg.id
                            colormap: colormap,
                            opacity: opacity
                        });
                    }

                    if (vol) {
                        nv.addVolume(vol);
                    }
                } catch (e) {
                    console.error("Failed to load seg:", seg.id, e);
                }
            }

            nv.updateGLVolume();
        };

        syncSegmentations();
    }, [nv, segmentations, image]); // Re-run if list changes

    return (
        <div className="w-full h-full bg-black rounded-lg overflow-hidden border border-slate-700 shadow-xl">
            <canvas ref={canvasRef} className="w-full h-full outline-none" />
        </div>
    );
}
