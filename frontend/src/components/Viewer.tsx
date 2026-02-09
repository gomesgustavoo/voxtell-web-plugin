import { useRef, useEffect, useState } from 'react';
import { Niivue, NVImage } from '@niivue/niivue';

interface ViewerProps {
    image?: File | string | null;
    segmentation?: File | string | null;
}

export default function Viewer({ image, segmentation }: ViewerProps) {
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
            // Cleanup if needed, though Niivue doesn't strictly require explicit detach usually
            // niivue.detachFromCanvas(); 
        };
    }, []);

    useEffect(() => {
        if (!nv || !image) return;

        const loadVolume = async () => {
            try {
                // Remove all existing volumes
                while (nv.volumes.length > 0) {
                    nv.removeVolume(nv.volumes[0]);
                }

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

    useEffect(() => {
        if (!nv || !segmentation) return;

        const loadSeg = async () => {
            // Remove existing overlays if any (assuming index 1+ are overlays)
            while (nv.volumes.length > 1) {
                nv.removeVolume(nv.volumes[nv.volumes.length - 1]);
            }

            try {
                let vol;
                if (typeof segmentation === 'string') {
                    vol = await NVImage.loadFromUrl({ url: segmentation, colormap: 'red', opacity: 0.5 });
                } else if (segmentation instanceof File) {
                    // @ts-ignore - NVImage types might be strict about file property
                    vol = await NVImage.loadFromFile({
                        file: segmentation,
                        name: segmentation.name,
                        colormap: 'red',
                        opacity: 0.5
                    });
                }

                if (vol) {
                    // @ts-ignore
                    nv.addVolume(vol);
                }
            } catch (e) {
                console.error("Failed to load segmentation:", e);
            }

            nv.updateGLVolume();
        };

        loadSeg();
    }, [nv, segmentation]);

    return (
        <div className="w-full h-full bg-black rounded-lg overflow-hidden border border-slate-700 shadow-xl">
            <canvas ref={canvasRef} className="w-full h-full outline-none" />
        </div>
    );
}
